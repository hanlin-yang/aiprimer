"""
Primer Design Agent Orchestrator

Implements a ReAct (Reasoning + Acting) agent loop for intelligent
primer design using LLM function calling with bio-computational tools.

Key Features:
- NLU parsing of natural language primer design requests
- Tool orchestration with Primer3 and BLAST services
- Self-reflection and automatic parameter adjustment
- Maximum 3 retry attempts with optimization feedback

Compliance: All bio-parameters computed via validated tools (no hallucination).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Union
import asyncio
import yaml
from pathlib import Path

from schemas.primer import (
    DesignConstraints,
    ErrorCode,
    PrimerDesignRequest,
    PrimerDesignResponse,
    PrimerPair,
    SpecificityResult,
    StructuralWarning,
    TaskType,
)
from schemas.variant import HighRiskVariantException
from services.bio_compute import (
    Primer3Wrapper,
    LocalBlastService,
    VariantChecker,
    audit_trail,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Tool Definitions for LLM Function Calling
# ============================================================================

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "design_primers",
            "description": "Design PCR primers for a given DNA template sequence using Primer3 algorithm. "
                          "Uses Nearest-Neighbor thermodynamics for accurate Tm calculation. "
                          "Returns ranked primer pairs with thermodynamic properties.",
            "parameters": {
                "type": "object",
                "properties": {
                    "template_sequence": {
                        "type": "string",
                        "description": "DNA template sequence (5' to 3'), ATCG characters only"
                    },
                    "task_type": {
                        "type": "string",
                        "enum": ["qPCR", "PCR", "LAMP", "NGS"],
                        "description": "Type of primer design task"
                    },
                    "tm_min": {
                        "type": "number",
                        "description": "Minimum melting temperature (°C), default 55"
                    },
                    "tm_max": {
                        "type": "number",
                        "description": "Maximum melting temperature (°C), default 60"
                    },
                    "product_size_min": {
                        "type": "integer",
                        "description": "Minimum amplicon size (bp)"
                    },
                    "product_size_max": {
                        "type": "integer",
                        "description": "Maximum amplicon size (bp)"
                    },
                    "gc_clamp": {
                        "type": "boolean",
                        "description": "Require GC clamp at 3' end"
                    },
                    "num_primers": {
                        "type": "integer",
                        "description": "Number of primer pairs to return (default 5)"
                    },
                    "target_start": {
                        "type": "integer",
                        "description": "Start position of target region within template"
                    },
                    "target_length": {
                        "type": "integer",
                        "description": "Length of target region"
                    }
                },
                "required": ["template_sequence"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_primer_specificity",
            "description": "Check primer specificity against a reference genome using BLAST. "
                          "Returns off-target binding sites and specificity score. "
                          "Essential for validating primer design quality.",
            "parameters": {
                "type": "object",
                "properties": {
                    "forward_primer": {
                        "type": "string",
                        "description": "Forward primer sequence (5' to 3')"
                    },
                    "reverse_primer": {
                        "type": "string",
                        "description": "Reverse primer sequence (5' to 3')"
                    },
                    "genome_database": {
                        "type": "string",
                        "enum": ["hg38", "hg19", "mm10", "mm39"],
                        "description": "Reference genome database for BLAST search"
                    }
                },
                "required": ["forward_primer", "reverse_primer"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_variant_conflicts",
            "description": "Check if primer binding sites contain SNPs or variants that may affect binding. "
                          "Queries dbSNP and gnomAD databases. Flags variants with MAF > 1%.",
            "parameters": {
                "type": "object",
                "properties": {
                    "forward_primer": {
                        "type": "string",
                        "description": "Forward primer sequence"
                    },
                    "reverse_primer": {
                        "type": "string",
                        "description": "Reverse primer sequence"
                    },
                    "chromosome": {
                        "type": "string",
                        "description": "Chromosome (e.g., 'chr1', 'chrX')"
                    },
                    "forward_position": {
                        "type": "integer",
                        "description": "Genomic start position of forward primer"
                    },
                    "reverse_position": {
                        "type": "integer",
                        "description": "Genomic start position of reverse primer"
                    },
                    "gene_symbol": {
                        "type": "string",
                        "description": "Target gene symbol for pre-filtering"
                    }
                },
                "required": ["forward_primer", "reverse_primer", "chromosome"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_thermodynamics",
            "description": "Calculate detailed thermodynamic properties for a primer pair. "
                          "Includes Tm, GC%, hairpin analysis, and dimer detection. "
                          "Uses Nearest-Neighbor model (SantaLucia 1998).",
            "parameters": {
                "type": "object",
                "properties": {
                    "forward_primer": {
                        "type": "string",
                        "description": "Forward primer sequence"
                    },
                    "reverse_primer": {
                        "type": "string",
                        "description": "Reverse primer sequence"
                    },
                    "probe_sequence": {
                        "type": "string",
                        "description": "Optional internal probe sequence for qPCR"
                    },
                    "na_concentration": {
                        "type": "number",
                        "description": "Sodium concentration (mM), default 50"
                    },
                    "mg_concentration": {
                        "type": "number",
                        "description": "Magnesium concentration (mM), default 1.5"
                    }
                },
                "required": ["forward_primer", "reverse_primer"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "optimize_primer_parameters",
            "description": "Suggest optimized parameters based on previous design failures. "
                          "Analyzes why design failed and recommends adjustments.",
            "parameters": {
                "type": "object",
                "properties": {
                    "failure_reason": {
                        "type": "string",
                        "description": "Description of why previous design failed"
                    },
                    "current_tm_range": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Current [tm_min, tm_max] range"
                    },
                    "current_product_size": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Current [size_min, size_max] range"
                    },
                    "gc_content": {
                        "type": "number",
                        "description": "Template GC content if known"
                    }
                },
                "required": ["failure_reason"]
            }
        }
    }
]


# ============================================================================
# System Prompt
# ============================================================================

SYSTEM_PROMPT = """You are a senior bioinformatics expert specializing in primer design for molecular diagnostics.

## CORE RESPONSIBILITIES
1. Analyze user primer design requests and extract key parameters
2. Design primers using validated computational tools (Primer3)
3. Verify primer specificity against reference genomes (BLAST)
4. Check for SNP/variant conflicts in primer binding sites
5. Provide actionable optimization recommendations

## CRITICAL RULES - YOU MUST FOLLOW THESE
1. **NEVER hallucinate thermodynamic values** - All Tm, GC%, and Delta G values MUST come from tool calculations
2. **ALWAYS check variants BEFORE finalizing design** - Variant conflicts can invalidate entire assays
3. **ALWAYS verify specificity** - Use BLAST to confirm primers are target-specific
4. **If design fails, ANALYZE the cause** - High GC content? Secondary structures? Provide specific remediation

## WORKFLOW (ReAct Loop)
For each primer design request:

1. **PARSE**: Extract target gene, task type, and constraints from user request
2. **CHECK VARIANTS**: First check target region for high-frequency variants
3. **DESIGN**: Call design_primers with appropriate parameters
4. **CALCULATE**: Get detailed thermodynamics for top candidates
5. **VERIFY**: BLAST check specificity of best primer pair
6. **REFLECT**: If issues found (off-target, dimer risk), adjust parameters and retry (max 3 attempts)
7. **REPORT**: Provide structured results with warnings and recommendations

## OPTIMIZATION STRATEGIES
When design fails or quality is suboptimal:
- **High GC region (>65%)**: Suggest adding 5% DMSO, use shorter primers, increase Tm threshold
- **Low GC region (<35%)**: Use longer primers, consider LNA modifications
- **Dimer formation (DeltaG < -9 kcal/mol)**: Shift primer position, redesign 3' end
- **Off-target hits**: Extend primers at 3' end, use more stringent Tm
- **Tm mismatch (>3°C)**: Adjust primer lengths to equilibrate Tm

## RESPONSE FORMAT
Always structure your analysis as:
1. Understanding of the request
2. Tool calls with reasoning
3. Interpretation of results
4. Warnings and recommendations
5. Final primer set with confidence assessment

You have access to the following tools:
- design_primers: Primer3-based design with Nearest-Neighbor thermodynamics
- check_primer_specificity: BLAST against reference genomes
- check_variant_conflicts: SNP/variant detection in primer sites
- calculate_thermodynamics: Detailed thermodynamic analysis
- optimize_primer_parameters: Parameter adjustment suggestions

Think step-by-step and show your reasoning at each stage."""


# ============================================================================
# Agent State Management
# ============================================================================

class AgentState(str, Enum):
    """Agent execution states."""
    IDLE = "idle"
    PARSING = "parsing_request"
    CHECKING_VARIANTS = "checking_variants"
    DESIGNING = "designing_primers"
    CALCULATING = "calculating_thermodynamics"
    VERIFYING = "verifying_specificity"
    REFLECTING = "reflecting_on_results"
    OPTIMIZING = "optimizing_parameters"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class OptimizationIteration:
    """Record of a single optimization attempt."""
    iteration: int
    state: AgentState
    parameters_used: Dict[str, Any]
    result_summary: str
    issues_found: List[str]
    adjustments_made: List[str]
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class AgentContext:
    """Maintains context across the ReAct loop."""
    request: PrimerDesignRequest
    state: AgentState = AgentState.IDLE
    iteration: int = 0
    max_iterations: int = 3

    # Results from each step
    variant_check_result: Optional[Dict] = None
    primer_candidates: List[PrimerPair] = field(default_factory=list)
    thermodynamics_result: Optional[Dict] = None
    specificity_result: Optional[Dict] = None

    # Optimization history
    optimization_history: List[OptimizationIteration] = field(default_factory=list)

    # Current best result
    best_primer_pair: Optional[PrimerPair] = None
    current_constraints: Optional[DesignConstraints] = None

    # Accumulated warnings
    warnings: List[StructuralWarning] = field(default_factory=list)
    error_codes: List[ErrorCode] = field(default_factory=list)

    # Messages for LLM context
    messages: List[Dict[str, Any]] = field(default_factory=list)

    def record_iteration(
        self,
        result_summary: str,
        issues: List[str],
        adjustments: List[str]
    ):
        """Record current optimization iteration."""
        self.optimization_history.append(OptimizationIteration(
            iteration=self.iteration,
            state=self.state,
            parameters_used={
                'tm_min': self.current_constraints.tm_min if self.current_constraints else None,
                'tm_max': self.current_constraints.tm_max if self.current_constraints else None,
                'product_size_min': self.current_constraints.product_size_min if self.current_constraints else None,
                'product_size_max': self.current_constraints.product_size_max if self.current_constraints else None,
            },
            result_summary=result_summary,
            issues_found=issues,
            adjustments_made=adjustments
        ))


# ============================================================================
# Prompt Version Manager
# ============================================================================

class PromptVersionManager:
    """
    Manages versioned prompts for regulatory compliance.

    Prompts are stored in YAML files with semantic versioning.
    Format: {agent-name}-v{major}.{minor}.{patch}-{status}
    """

    def __init__(self, prompts_dir: str = "prompts"):
        self.prompts_dir = Path(prompts_dir)
        self.versions: Dict[str, Dict] = {}
        self._load_versions()

    def _load_versions(self):
        """Load all prompt versions from YAML files."""
        versions_file = self.prompts_dir / "versions.yaml"
        if versions_file.exists():
            with open(versions_file) as f:
                self.versions = yaml.safe_load(f) or {}

    def get_prompt(
        self,
        agent_name: str,
        version: Optional[str] = None
    ) -> str:
        """
        Get a specific prompt version.

        Args:
            agent_name: Name of the agent (e.g., 'primer-agent')
            version: Semantic version (e.g., 'v1.2.0') or 'latest'

        Returns:
            Prompt string
        """
        if agent_name not in self.versions:
            logger.warning(f"Agent {agent_name} not found, using default")
            return SYSTEM_PROMPT

        agent_versions = self.versions[agent_name]

        if version is None or version == 'latest':
            # Get the latest stable version
            stable_versions = [
                v for v in agent_versions.keys()
                if 'stable' in v or 'release' in v
            ]
            if stable_versions:
                version = sorted(stable_versions)[-1]
            else:
                version = list(agent_versions.keys())[-1]

        if version not in agent_versions:
            raise ValueError(f"Version {version} not found for {agent_name}")

        return agent_versions[version]['prompt']

    def list_versions(self, agent_name: str) -> List[str]:
        """List all available versions for an agent."""
        if agent_name not in self.versions:
            return []
        return list(self.versions[agent_name].keys())


# ============================================================================
# LLM Client Interface
# ============================================================================

class LLMClient:
    """
    Abstract LLM client for function calling.

    Supports both OpenAI and Anthropic APIs.
    """

    def __init__(
        self,
        provider: str = "anthropic",
        model: str = "claude-sonnet-4-20250514",
        api_key: Optional[str] = None
    ):
        self.provider = provider
        self.model = model
        self.api_key = api_key

    async def chat_completion(
        self,
        messages: List[Dict],
        tools: List[Dict],
        system_prompt: str
    ) -> Dict[str, Any]:
        """
        Send chat completion request with function calling.

        Returns response with potential tool calls.
        """
        if self.provider == "anthropic":
            return await self._anthropic_completion(messages, tools, system_prompt)
        elif self.provider == "openai":
            return await self._openai_completion(messages, tools, system_prompt)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    async def _anthropic_completion(
        self,
        messages: List[Dict],
        tools: List[Dict],
        system_prompt: str
    ) -> Dict[str, Any]:
        """Anthropic Claude API call."""
        try:
            import anthropic
        except ImportError:
            logger.error("anthropic package not installed")
            raise

        client = anthropic.AsyncAnthropic(api_key=self.api_key)

        # Convert tools to Anthropic format
        anthropic_tools = []
        for tool in tools:
            if tool["type"] == "function":
                anthropic_tools.append({
                    "name": tool["function"]["name"],
                    "description": tool["function"]["description"],
                    "input_schema": tool["function"]["parameters"]
                })

        response = await client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=system_prompt,
            tools=anthropic_tools,
            messages=messages
        )

        return self._parse_anthropic_response(response)

    async def _openai_completion(
        self,
        messages: List[Dict],
        tools: List[Dict],
        system_prompt: str
    ) -> Dict[str, Any]:
        """OpenAI API call."""
        try:
            import openai
        except ImportError:
            logger.error("openai package not installed")
            raise

        client = openai.AsyncOpenAI(api_key=self.api_key)

        # Prepend system message
        full_messages = [{"role": "system", "content": system_prompt}] + messages

        response = await client.chat.completions.create(
            model=self.model,
            messages=full_messages,
            tools=tools,
            tool_choice="auto"
        )

        return self._parse_openai_response(response)

    def _parse_anthropic_response(self, response) -> Dict[str, Any]:
        """Parse Anthropic response format."""
        result = {
            "content": "",
            "tool_calls": [],
            "stop_reason": response.stop_reason
        }

        for block in response.content:
            if hasattr(block, 'text'):
                result["content"] += block.text
            elif hasattr(block, 'type') and block.type == 'tool_use':
                result["tool_calls"].append({
                    "id": block.id,
                    "name": block.name,
                    "arguments": block.input
                })

        return result

    def _parse_openai_response(self, response) -> Dict[str, Any]:
        """Parse OpenAI response format."""
        message = response.choices[0].message

        result = {
            "content": message.content or "",
            "tool_calls": [],
            "stop_reason": response.choices[0].finish_reason
        }

        if message.tool_calls:
            for tc in message.tool_calls:
                result["tool_calls"].append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": json.loads(tc.function.arguments)
                })

        return result


# ============================================================================
# Tool Executor
# ============================================================================

class ToolExecutor:
    """
    Executes bio-computational tools called by the LLM.

    All tool results are computed - never hallucinated.
    """

    def __init__(self):
        self.primer3 = Primer3Wrapper()
        self.blast = LocalBlastService()
        self.variant_checker = VariantChecker()

    async def execute(
        self,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a tool and return results.

        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments from LLM

        Returns:
            Tool execution result
        """
        start_time = time.time()

        try:
            if tool_name == "design_primers":
                result = await self._execute_design_primers(arguments)
            elif tool_name == "check_primer_specificity":
                result = await self._execute_specificity_check(arguments)
            elif tool_name == "check_variant_conflicts":
                result = await self._execute_variant_check(arguments)
            elif tool_name == "calculate_thermodynamics":
                result = await self._execute_thermodynamics(arguments)
            elif tool_name == "optimize_primer_parameters":
                result = await self._execute_optimization(arguments)
            else:
                raise ValueError(f"Unknown tool: {tool_name}")

            result["_execution_time_ms"] = int((time.time() - start_time) * 1000)
            result["_success"] = True
            return result

        except Exception as e:
            logger.error(f"Tool execution failed: {tool_name} - {e}")
            return {
                "_success": False,
                "_error": str(e),
                "_execution_time_ms": int((time.time() - start_time) * 1000)
            }

    async def _execute_design_primers(
        self,
        args: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute primer design."""
        # Build constraints
        constraints = DesignConstraints(
            tm_min=args.get("tm_min", 55.0),
            tm_max=args.get("tm_max", 60.0),
            product_size_min=args.get("product_size_min", 75),
            product_size_max=args.get("product_size_max", 200),
            gc_clamp=args.get("gc_clamp", True)
        )

        # Parse task type
        task_type = TaskType(args.get("task_type", "PCR"))

        # Get target region if specified
        target_region = None
        if "target_start" in args and "target_length" in args:
            target_region = (args["target_start"], args["target_length"])

        # Design primers
        primer_pairs = self.primer3.design_primers(
            template_sequence=args["template_sequence"],
            constraints=constraints,
            target_region=target_region,
            task_type=task_type,
            num_return=args.get("num_primers", 5)
        )

        # Convert to serializable format
        return {
            "num_pairs_found": len(primer_pairs),
            "primer_pairs": [pp.model_dump() for pp in primer_pairs],
            "task_type": task_type.value,
            "constraints_used": constraints.model_dump()
        }

    async def _execute_specificity_check(
        self,
        args: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute BLAST specificity check."""
        genome_db = args.get("genome_database", "hg38")

        # Check both primers
        result = await self.blast.check_primer_pair_specificity(
            forward_seq=args["forward_primer"],
            reverse_seq=args["reverse_primer"],
            genome_db=genome_db
        )

        return {
            "forward_specificity": result["forward_specificity"].model_dump(),
            "reverse_specificity": result["reverse_specificity"].model_dump(),
            "pair_is_specific": result["pair_specific"],
            "recommendation": result["recommendation"],
            "database_used": genome_db
        }

    async def _execute_variant_check(
        self,
        args: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute variant conflict check."""
        try:
            result = await self.variant_checker.validate_primer_pair(
                forward_seq=args["forward_primer"],
                reverse_seq=args["reverse_primer"],
                forward_start=args.get("forward_position", 0),
                reverse_start=args.get("reverse_position", 0),
                chromosome=args["chromosome"],
                gene_symbol=args.get("gene_symbol")
            )

            return {
                "is_valid": result["is_valid"],
                "num_conflicts": len(result["conflicts"]),
                "conflicts": [
                    {
                        "primer": c.primer_type,
                        "variant": c.variant.rsid,
                        "maf": c.variant.global_maf,
                        "severity": c.impact_severity,
                        "recommendation": c.recommendation
                    }
                    for c in result["conflicts"]
                ],
                "risk_score": result["risk_score"],
                "recommendations": result["recommendations"]
            }

        except HighRiskVariantException as e:
            return {
                "is_valid": False,
                "critical_error": True,
                "message": str(e),
                "risk_score": e.risk_score,
                "recommendations": e.recommendations
            }

    async def _execute_thermodynamics(
        self,
        args: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute thermodynamic calculations."""
        # Create wrapper with specified salt concentrations
        wrapper = Primer3Wrapper(
            mv_conc=args.get("na_concentration", 50.0),
            dv_conc=args.get("mg_concentration", 1.5)
        )

        result = wrapper.calculate_thermodynamics(
            forward_seq=args["forward_primer"],
            reverse_seq=args["reverse_primer"],
            probe_seq=args.get("probe_sequence")
        )

        return result

    async def _execute_optimization(
        self,
        args: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate optimization suggestions based on failure analysis."""
        failure_reason = args.get("failure_reason", "").lower()
        current_tm = args.get("current_tm_range", [55, 60])
        current_size = args.get("current_product_size", [75, 200])
        gc_content = args.get("gc_content")

        suggestions = []
        new_params = {}

        # Analyze failure and suggest adjustments
        if "gc" in failure_reason and "high" in failure_reason:
            suggestions.append(
                "High GC content detected. Consider: "
                "(1) Add 5% DMSO to reaction, "
                "(2) Use shorter primers (18-20 bp), "
                "(3) Increase Tm threshold"
            )
            new_params["tm_min"] = current_tm[0] + 2
            new_params["tm_max"] = current_tm[1] + 3
            new_params["add_dmso"] = "5%"

        if "dimer" in failure_reason:
            suggestions.append(
                "Dimer formation risk. Consider: "
                "(1) Shift primer positions, "
                "(2) Reduce primer length at 3' end, "
                "(3) Check for complementary 3' sequences"
            )
            new_params["max_self_complementarity"] = 6
            new_params["redesign_3_prime"] = True

        if "off-target" in failure_reason or "specificity" in failure_reason:
            suggestions.append(
                "Poor specificity detected. Consider: "
                "(1) Extend primers at 3' end by 2-3 bases, "
                "(2) Increase Tm minimum, "
                "(3) Use more unique target region"
            )
            new_params["tm_min"] = current_tm[0] + 3
            new_params["primer_length_min"] = 20

        if "tm" in failure_reason and "mismatch" in failure_reason:
            suggestions.append(
                "Tm mismatch between primers. Consider: "
                "(1) Adjust primer lengths to equilibrate Tm, "
                "(2) Target Tm within 1°C difference"
            )
            new_params["tm_difference_max"] = 1.0

        if "variant" in failure_reason or "snp" in failure_reason:
            suggestions.append(
                "SNP conflict in primer binding site. Consider: "
                "(1) Shift primer position to avoid variant, "
                "(2) Design degenerate primer if variant is frequent, "
                "(3) Use alternative binding region"
            )
            new_params["avoid_variants"] = True

        if "no primers" in failure_reason or "no valid" in failure_reason:
            suggestions.append(
                "No valid primers found. Consider: "
                "(1) Relax Tm constraints, "
                "(2) Expand product size range, "
                "(3) Allow lower GC clamp stringency"
            )
            new_params["tm_min"] = current_tm[0] - 3
            new_params["tm_max"] = current_tm[1] + 3
            new_params["product_size_min"] = max(50, current_size[0] - 25)
            new_params["product_size_max"] = current_size[1] + 50
            new_params["gc_clamp"] = False

        return {
            "analysis": failure_reason,
            "suggestions": suggestions,
            "recommended_parameters": new_params,
            "confidence": 0.8 if suggestions else 0.5
        }


# ============================================================================
# Primer Design Agent
# ============================================================================

class PrimerAgent:
    """
    ReAct-based Primer Design Agent.

    Implements a reasoning-action loop for intelligent primer design
    with self-reflection and automatic optimization.
    """

    MAX_ITERATIONS = 3

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        prompt_version: str = "latest"
    ):
        self.llm_client = llm_client or LLMClient()
        self.tool_executor = ToolExecutor()
        self.prompt_manager = PromptVersionManager()
        self.prompt_version = prompt_version

        # Get versioned prompt
        try:
            self.system_prompt = self.prompt_manager.get_prompt(
                "primer-agent",
                prompt_version
            )
        except Exception:
            self.system_prompt = SYSTEM_PROMPT

    async def run_design_loop(
        self,
        request: PrimerDesignRequest
    ) -> PrimerDesignResponse:
        """
        Execute the ReAct design loop.

        Steps:
        1. Parse user request (NLU)
        2. Check variants in target region
        3. Design primers using Primer3
        4. Verify specificity with BLAST
        5. If issues found, reflect and retry (max 3 times)

        Args:
            request: Primer design request

        Returns:
            Complete primer design response
        """
        start_time = time.time()

        # Initialize context
        context = AgentContext(
            request=request,
            current_constraints=request.constraints,
            max_iterations=self.MAX_ITERATIONS
        )

        # Initial user message
        user_message = self._format_user_request(request)
        context.messages.append({"role": "user", "content": user_message})

        # ReAct Loop
        while context.iteration < context.max_iterations:
            context.iteration += 1
            logger.info(f"Design iteration {context.iteration}/{context.max_iterations}")

            try:
                # Get LLM response with tool calls
                context.state = AgentState.PARSING
                response = await self.llm_client.chat_completion(
                    messages=context.messages,
                    tools=TOOL_DEFINITIONS,
                    system_prompt=self.system_prompt
                )

                # Process tool calls
                if response["tool_calls"]:
                    tool_results = await self._process_tool_calls(
                        response["tool_calls"],
                        context
                    )

                    # Add assistant message and tool results
                    context.messages.append({
                        "role": "assistant",
                        "content": response["content"],
                        "tool_calls": response["tool_calls"]
                    })

                    for tool_call, result in zip(response["tool_calls"], tool_results):
                        context.messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "content": json.dumps(result)
                        })

                    # Check if we should continue or stop
                    should_continue, issues = self._evaluate_results(context)

                    if not should_continue:
                        # Design successful or no more improvements possible
                        context.state = AgentState.COMPLETED
                        break

                    # Record iteration and continue optimization
                    context.record_iteration(
                        result_summary=f"Issues found: {len(issues)}",
                        issues=issues,
                        adjustments_made=["Requesting optimization suggestions"]
                    )

                else:
                    # No tool calls - agent has concluded
                    context.messages.append({
                        "role": "assistant",
                        "content": response["content"]
                    })
                    context.state = AgentState.COMPLETED
                    break

            except Exception as e:
                logger.error(f"Design loop error: {e}")
                context.state = AgentState.FAILED
                context.error_codes.append(ErrorCode.ERR_PRIMER3_FAILURE)
                break

        # Build response
        computation_time = int((time.time() - start_time) * 1000)
        return self._build_response(context, computation_time)

    def _format_user_request(self, request: PrimerDesignRequest) -> str:
        """Format the primer design request as a user message."""
        parts = [
            f"Please design {request.task_type.value} primers for gene {request.target_gene}.",
        ]

        if request.sequence_template:
            parts.append(f"\nTemplate sequence ({len(request.sequence_template)} bp):")
            # Show first and last 50 bp for long sequences
            if len(request.sequence_template) > 120:
                parts.append(f"{request.sequence_template[:50]}...{request.sequence_template[-50:]}")
            else:
                parts.append(request.sequence_template)

        parts.append(f"\nConstraints:")
        parts.append(f"- Tm range: {request.constraints.tm_min}-{request.constraints.tm_max}°C")
        parts.append(f"- Product size: {request.constraints.product_size_min}-{request.constraints.product_size_max} bp")
        parts.append(f"- GC clamp: {'Yes' if request.constraints.gc_clamp else 'No'}")

        if request.check_variants:
            parts.append(f"- Check for variants with MAF > {request.variant_maf_threshold:.1%}")

        if request.check_specificity:
            parts.append(f"- Verify specificity against {request.genome_database}")

        return "\n".join(parts)

    async def _process_tool_calls(
        self,
        tool_calls: List[Dict],
        context: AgentContext
    ) -> List[Dict]:
        """Execute all tool calls and collect results."""
        results = []

        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            arguments = tool_call["arguments"]

            logger.info(f"Executing tool: {tool_name}")

            # Update state based on tool
            if tool_name == "design_primers":
                context.state = AgentState.DESIGNING
            elif tool_name == "check_primer_specificity":
                context.state = AgentState.VERIFYING
            elif tool_name == "check_variant_conflicts":
                context.state = AgentState.CHECKING_VARIANTS
            elif tool_name == "calculate_thermodynamics":
                context.state = AgentState.CALCULATING
            elif tool_name == "optimize_primer_parameters":
                context.state = AgentState.OPTIMIZING

            result = await self.tool_executor.execute(tool_name, arguments)
            results.append(result)

            # Update context with results
            self._update_context_from_result(tool_name, result, context)

        return results

    def _update_context_from_result(
        self,
        tool_name: str,
        result: Dict,
        context: AgentContext
    ):
        """Update agent context based on tool results."""
        if tool_name == "design_primers" and result.get("_success"):
            # Store primer candidates
            if result.get("primer_pairs"):
                from schemas.primer import PrimerPair
                context.primer_candidates = [
                    PrimerPair(**pp) for pp in result["primer_pairs"]
                ]
                if context.primer_candidates:
                    context.best_primer_pair = context.primer_candidates[0]

        elif tool_name == "check_primer_specificity":
            context.specificity_result = result

            # Add warnings for specificity issues
            if not result.get("pair_is_specific", True):
                context.warnings.append(StructuralWarning(
                    code=ErrorCode.ERR_OFF_TARGET,
                    severity="warning",
                    message="Off-target binding detected",
                    suggestion=result.get("recommendation", "")
                ))

        elif tool_name == "check_variant_conflicts":
            context.variant_check_result = result

            if result.get("critical_error"):
                context.error_codes.append(ErrorCode.ERR_SNP_CONFLICT)
                context.warnings.append(StructuralWarning(
                    code=ErrorCode.ERR_SNP_CONFLICT,
                    severity="critical",
                    message=result.get("message", "High-risk variant detected"),
                    suggestion="; ".join(result.get("recommendations", []))
                ))

        elif tool_name == "calculate_thermodynamics":
            context.thermodynamics_result = result

            # Add warnings from thermodynamic analysis
            for warning in result.get("warnings", []):
                context.warnings.append(StructuralWarning(
                    code=warning.get("code", ErrorCode.ERR_DIMER_RISK),
                    severity="warning",
                    message=warning.get("message", ""),
                    suggestion=warning.get("suggestion", "")
                ))

    def _evaluate_results(
        self,
        context: AgentContext
    ) -> tuple[bool, List[str]]:
        """
        Evaluate current results and decide whether to continue optimization.

        Returns:
            Tuple of (should_continue, list_of_issues)
        """
        issues = []

        # Check if we have valid primers
        if not context.primer_candidates:
            issues.append("No valid primer pairs generated")
            return True, issues

        # Check specificity
        if context.specificity_result:
            if not context.specificity_result.get("pair_is_specific", True):
                issues.append("Poor primer specificity")

        # Check for variant conflicts
        if context.variant_check_result:
            if not context.variant_check_result.get("is_valid", True):
                issues.append("Variant conflicts in primer binding sites")

        # Check thermodynamics
        if context.thermodynamics_result:
            if context.thermodynamics_result.get("requires_optimization"):
                issues.append("Thermodynamic optimization required")

            # Check Tm difference
            tm_diff = context.thermodynamics_result.get("pair", {}).get("tm_difference", 0)
            if tm_diff > 3.0:
                issues.append(f"Tm difference too large ({tm_diff:.1f}°C)")

        # Check for critical warnings
        critical_warnings = [w for w in context.warnings if w.severity == "critical"]
        if critical_warnings:
            issues.append("Critical warnings present")

        # Decide whether to continue
        should_continue = len(issues) > 0 and context.iteration < context.max_iterations

        return should_continue, issues

    def _build_response(
        self,
        context: AgentContext,
        computation_time_ms: int
    ) -> PrimerDesignResponse:
        """Build the final primer design response."""
        # Determine status
        if context.state == AgentState.FAILED:
            status = "failed"
        elif context.primer_candidates:
            status = "success" if not context.warnings else "partial"
        else:
            status = "failed"

        # Generate AI suggestion based on context
        ai_suggestion = self._generate_ai_suggestion(context)

        # Build optimization history for traceability
        optimization_history = [
            {
                "iteration": h.iteration,
                "state": h.state.value,
                "parameters": h.parameters_used,
                "result": h.result_summary,
                "issues": h.issues_found,
                "adjustments": h.adjustments_made,
                "timestamp": h.timestamp.isoformat()
            }
            for h in context.optimization_history
        ]

        # Build response
        response = PrimerDesignResponse(
            request_id=context.request.request_id,
            status=status,
            task_type=context.request.task_type,
            target_gene=context.request.target_gene,
            primer_pairs=context.primer_candidates,
            best_pair=context.best_primer_pair,
            specificity_results=(
                SpecificityResult(**context.specificity_result["forward_specificity"])
                if context.specificity_result else None
            ),
            structural_warnings=context.warnings,
            variant_warnings=(
                context.variant_check_result.get("conflicts", [])
                if context.variant_check_result else []
            ),
            ai_suggestion=ai_suggestion,
            optimization_history=optimization_history,
            error_codes=context.error_codes,
            computation_time_ms=computation_time_ms,
            primer3_version=Primer3Wrapper.PRIMER3_VERSION,
            blast_database=context.request.genome_database
        )

        # Log to audit trail
        audit_trail.log_operation(
            operation="primer_design_complete",
            input_data={
                "request_id": context.request.request_id,
                "target_gene": context.request.target_gene
            },
            output_data={
                "status": status,
                "num_pairs": len(context.primer_candidates),
                "num_warnings": len(context.warnings)
            },
            parameters={"iterations": context.iteration},
            computation_time_ms=computation_time_ms,
            success=status != "failed",
            user_id=context.request.user_id
        )

        return response

    def _generate_ai_suggestion(self, context: AgentContext) -> str:
        """Generate actionable AI suggestion based on analysis."""
        suggestions = []

        if not context.primer_candidates:
            suggestions.append(
                "No valid primers could be designed with current constraints. "
                "Consider relaxing Tm range or product size limits."
            )
            return " ".join(suggestions)

        # Check Tm difference
        if context.best_primer_pair:
            tm_diff = context.best_primer_pair.tm_difference
            if tm_diff > 3.0:
                suggestions.append(
                    f"Tm difference between primers is {tm_diff:.1f}°C, which exceeds the recommended 3°C maximum. "
                    "Consider adjusting primer lengths to equilibrate melting temperatures."
                )
            elif tm_diff > 1.0:
                suggestions.append(
                    f"Tm difference ({tm_diff:.1f}°C) is acceptable but could be optimized for better performance."
                )

        # Check dimer risk
        if context.best_primer_pair and context.best_primer_pair.requires_optimization:
            suggestions.append(
                f"Dimer formation risk detected (ΔG = {context.best_primer_pair.dimer_delta_g:.1f} kcal/mol). "
                "Consider adding 5% DMSO to reaction or hot-start polymerase to minimize dimer amplification."
            )

        # Specificity suggestions
        if context.specificity_result:
            if not context.specificity_result.get("pair_is_specific", True):
                suggestions.append(
                    "Primers show potential off-target binding. "
                    "Verify with in-silico PCR or use touchdown PCR protocol to improve specificity."
                )

        # Variant suggestions
        if context.variant_check_result:
            risk_score = context.variant_check_result.get("risk_score", 0)
            if risk_score > 0.5:
                suggestions.append(
                    "Significant variant conflicts detected in primer binding sites. "
                    "Consider population-specific validation or degenerate primer design."
                )

        # GC content suggestions
        if context.thermodynamics_result:
            fwd_gc = context.thermodynamics_result.get("forward", {}).get("gc_percent", 50)
            rev_gc = context.thermodynamics_result.get("reverse", {}).get("gc_percent", 50)

            if fwd_gc > 65 or rev_gc > 65:
                suggestions.append(
                    "High GC content detected. Adding 5% DMSO or using GC-enhancer buffer is recommended."
                )
            elif fwd_gc < 35 or rev_gc < 35:
                suggestions.append(
                    "Low GC content detected. Ensure primers have adequate 3' stability."
                )

        if not suggestions:
            suggestions.append(
                "Primer pair meets all design criteria. "
                "Recommended for experimental validation."
            )

        return " ".join(suggestions)


# ============================================================================
# Convenience Functions
# ============================================================================

async def design_primers(
    target_gene: str,
    sequence_template: Optional[str] = None,
    task_type: str = "PCR",
    **kwargs
) -> PrimerDesignResponse:
    """
    Convenience function for primer design.

    Args:
        target_gene: Target gene symbol
        sequence_template: DNA template sequence
        task_type: Type of primer design task
        **kwargs: Additional constraints

    Returns:
        PrimerDesignResponse
    """
    # Build constraints
    constraints = DesignConstraints(
        tm_min=kwargs.get("tm_min", 55.0),
        tm_max=kwargs.get("tm_max", 60.0),
        product_size_min=kwargs.get("product_size_min", 75),
        product_size_max=kwargs.get("product_size_max", 200),
        gc_clamp=kwargs.get("gc_clamp", True)
    )

    # Build request
    request = PrimerDesignRequest(
        target_gene=target_gene,
        sequence_template=sequence_template,
        task_type=TaskType(task_type),
        constraints=constraints,
        check_specificity=kwargs.get("check_specificity", True),
        check_variants=kwargs.get("check_variants", True),
        genome_database=kwargs.get("genome_database", "hg38")
    )

    # Run agent
    agent = PrimerAgent()
    return await agent.run_design_loop(request)
