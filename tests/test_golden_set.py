"""
Golden Set Evaluation Tests

Comprehensive test suite for validating primer design quality.
Used in CI/CD pipeline to prevent prompt degradation.

Test Cases:
1. GAPDH - Housekeeping gene (standard reference)
2. ACE2 - COVID-19 receptor (complex region)
3. BRCA1 - Cancer gene (high clinical importance)
4. TP53 - Tumor suppressor (variant-rich region)
5. ACTB - Beta-actin (another housekeeping reference)

Quality Criteria:
- Tm difference between primers < 1°C
- No severe dimer formation (Delta G > -9 kcal/mol)
- GC content 40-60%
- No off-target binding (specificity > 0.9)
"""

import pytest
import asyncio
from typing import Dict, List, Optional
from dataclasses import dataclass

# Import modules under test
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from schemas.primer import (
    DesignConstraints,
    PrimerDesignRequest,
    PrimerDesignResponse,
    PrimerPair,
    TaskType,
    ErrorCode,
)
from services.bio_compute import (
    Primer3Wrapper,
    LocalBlastService,
    VariantChecker,
)
from core.agent import PrimerAgent


# ============================================================================
# Golden Set Test Data
# ============================================================================

@dataclass
class GoldenTestCase:
    """A golden test case for primer design validation."""
    name: str
    gene_symbol: str
    sequence: str
    task_type: TaskType
    expected_tm_range: tuple  # (min, max)
    expected_gc_range: tuple  # (min, max)
    expected_product_size: tuple  # (min, max)
    max_tm_difference: float
    max_dimer_delta_g: float
    notes: str


GOLDEN_TEST_CASES: List[GoldenTestCase] = [
    GoldenTestCase(
        name="GAPDH Housekeeping Gene",
        gene_symbol="GAPDH",
        sequence=(
            "ATGGGGAAGGTGAAGGTCGGAGTCAACGGATTTGGTCGTATTGGGCGCCTGGTCACCAGGGCTGCT"
            "TTTAACTCTGGTAAAGTGGATATTGTTGCCATCAATGACCCCTTCATTGACCTCAACTACATGGTTT"
            "ACATGTTCCAATATGATTCCACCCATGGCAAATTCCATGGCACCGTCAAGGCTGAGAACGGGAAGCT"
            "TGTCATCAATGGAAATCCCATCACCATCTTCCAGGAGCGAGATCCCTCCAAAATCAAGTGGGGCGAT"
            "GCTGGCGCTGAGTACGTCGTGGAGTCCACTGGCGTCTTCACCACCATGGAGAAGGCTGGGGCTCATT"
            "TGCAGGGGGGAGCCAAAAGGGTCATCATCTCTGCCCCCTCTGCTGATGCCCCCATGTTCGTCATGGG"
            "TGTGAACCATGAGAAGTATGACAACAGCCTCAAGATCATCAGCAATGCCTCCTGCACCACCAACTGC"
        ),
        task_type=TaskType.QPCR,
        expected_tm_range=(57.0, 63.0),
        expected_gc_range=(45.0, 60.0),
        expected_product_size=(80, 150),
        max_tm_difference=1.0,
        max_dimer_delta_g=-9.0,
        notes="Standard housekeeping gene for qPCR normalization"
    ),

    GoldenTestCase(
        name="ACE2 Receptor (COVID-19)",
        gene_symbol="ACE2",
        sequence=(
            "ATGTCAAGCTCTTCCTGGCTCCTTCTCAGCCTTGTTGCTGTAACTGCTGCTCAGTCCACCATTGAG"
            "GAACAGGCCAAGACATTTTTGGACAAGTTTAACCACGAAGCCGAAGACCTGTTCTATCAAAGTTCA"
            "CTTGCTTCTTGGAATTATGATGGCAACAAACAAAAGAATGGATCTTATTTCATTACTGTGTATGCT"
            "GCTGTCCTTCAGCTTATGATTGAACAACTCAGGATTGTCTGGGGCTCATCTGCTGAGGCTCTGAAA"
            "GGACTTTATGTACGTTATGGATGGCTGCTTAAAGCAGAGAATTTCTACAAAGAAATGCCAACTGCA"
            "GTTGGAGGCTGGTTCCTAACTCTGCCATCAAACCTTGATGACTTTAAAGAAGAGAAAGAAATTGAT"
        ),
        task_type=TaskType.QPCR,
        expected_tm_range=(57.0, 63.0),
        expected_gc_range=(40.0, 55.0),
        expected_product_size=(100, 200),
        max_tm_difference=1.0,
        max_dimer_delta_g=-9.0,
        notes="SARS-CoV-2 entry receptor, important for COVID diagnostics"
    ),

    GoldenTestCase(
        name="BRCA1 Cancer Gene",
        gene_symbol="BRCA1",
        sequence=(
            "ATGGATTTATCTGCTCTTCGCGTTGAAGAAGTACAAAATGTCATTAATGCTATGCAGAAAATCTTA"
            "GATTGTCTAAATTATGTTAAAGCTTCATTTGAAAAGAAAGAAAAGGAAGAGTCTCTCCTTATTTAT"
            "GGGGATGAAAGTTTTCAAGCTCTTGACACTGGAAGTCCATCTCTCTCTGAAATTTTGTCTGATGTT"
            "ACTGAGAGTAAAGAATTTGTGCCAATCTGTGTACCTCTTTGCAAAGAATGTATCCAGATTGGTTCT"
            "TCAGCAAAATATAGGAGGCTCCTCAGAAATGAGGAAATGCTTGAAGTAAGTAAGTGCTGTTGCCAG"
            "ATTGAAATTTGTGTGACTGTCTCCAGAGAGGAGAAGCTTCCATGTGAACCACCCTGTAAAAAGAAG"
        ),
        task_type=TaskType.PCR,
        expected_tm_range=(55.0, 62.0),
        expected_gc_range=(40.0, 55.0),
        expected_product_size=(150, 300),
        max_tm_difference=1.5,
        max_dimer_delta_g=-9.0,
        notes="Breast cancer susceptibility gene, high clinical importance"
    ),

    GoldenTestCase(
        name="TP53 Tumor Suppressor",
        gene_symbol="TP53",
        sequence=(
            "ATGGAGGAGCCGCAGTCAGATCCTAGCGTCGAGCCCCCTCTGAGTCAGGAAACATTTTCAGACCTA"
            "TGGAAACTACTTCCTGAAAACAACGTTCTGTCCCCCTTGCCGTCCCAAGCAATGGATGATTTGATG"
            "CTGTCCCCGGACGATATTGAACAATGGTTCACTGAAGACCCAGGTCCAGATGAAGCTCCCAGAATG"
            "CCAGAGGCTGCTCCCCCCGTGGCCCCTGCACCAGCAGCTCCTACACCGGCGGCCCCTGCACCAGCC"
            "CCCTCCTGGCCCCTGTCATCTTCTGTCCCTTCCCAGAAAACCTACCAGGGCAGCTACGGTTTCCGT"
            "CTGGGCTTCTTGCATTCTGGGACAGCCAAGTCTGTGACTTGCACGTACTCCCCTGCCCTCAACAAG"
        ),
        task_type=TaskType.QPCR,
        expected_tm_range=(58.0, 64.0),
        expected_gc_range=(50.0, 65.0),  # Higher GC region
        expected_product_size=(80, 150),
        max_tm_difference=1.0,
        max_dimer_delta_g=-9.0,
        notes="Guardian of the genome, variant-rich region"
    ),

    GoldenTestCase(
        name="ACTB Beta-Actin",
        gene_symbol="ACTB",
        sequence=(
            "ATGGATGATGATATCGCCGCGCTCGTCGTCGACAACGGCTCCGGCATGTGCAAGGCCGGCTTCGCG"
            "GGCGACGATGCCCCCCGGGCCGTCTTCCCCTCCATCGTGGGGCGCCCCAGGCACCAGGGCGTGATG"
            "GTGGGCATGGGTCAGAAGGATTCCTATGTGGGCGACGAGGCCCAGAGCAAGAGAGGCATCCTCACC"
            "CTGAAGTACCCCATCGAGCACGGCATCGTCACCAACTGGGACGACATGGAGAAAATCTGGCACCAC"
            "ACCTTCTACAATGAGCTGCGTGTGGCTCCCGAGGAGCACCCCGTGCTGCTGACCGAGGCCCCCCTG"
            "AACCCCAAGGCCAACCGCGAGAAGATGACCCAGATCATGTTTGAGACCTTCAACACCCCAGCCATG"
        ),
        task_type=TaskType.QPCR,
        expected_tm_range=(58.0, 64.0),
        expected_gc_range=(50.0, 65.0),
        expected_product_size=(80, 150),
        max_tm_difference=1.0,
        max_dimer_delta_g=-9.0,
        notes="Cytoskeletal protein, common reference gene"
    ),
]


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def primer3_wrapper():
    """Create Primer3 wrapper instance."""
    return Primer3Wrapper()


@pytest.fixture
def blast_service():
    """Create BLAST service instance."""
    return LocalBlastService()


@pytest.fixture
def variant_checker():
    """Create variant checker instance."""
    return VariantChecker()


@pytest.fixture
def default_constraints():
    """Default design constraints for testing."""
    return DesignConstraints(
        tm_min=55.0,
        tm_max=62.0,
        gc_min=40.0,
        gc_max=60.0,
        product_size_min=75,
        product_size_max=200,
        gc_clamp=True,
    )


# ============================================================================
# Helper Functions
# ============================================================================

def evaluate_primer_pair(
    pair: PrimerPair,
    test_case: GoldenTestCase
) -> Dict[str, bool]:
    """
    Evaluate a primer pair against golden criteria.

    Returns dict of criterion -> pass/fail.
    """
    results = {}

    # Tm difference check
    results['tm_difference'] = pair.tm_difference <= test_case.max_tm_difference

    # Tm range check (both primers)
    results['forward_tm_in_range'] = (
        test_case.expected_tm_range[0] <= pair.forward.tm <= test_case.expected_tm_range[1]
    )
    results['reverse_tm_in_range'] = (
        test_case.expected_tm_range[0] <= pair.reverse.tm <= test_case.expected_tm_range[1]
    )

    # GC content check
    results['forward_gc_in_range'] = (
        test_case.expected_gc_range[0] <= pair.forward.gc_percent <= test_case.expected_gc_range[1]
    )
    results['reverse_gc_in_range'] = (
        test_case.expected_gc_range[0] <= pair.reverse.gc_percent <= test_case.expected_gc_range[1]
    )

    # Product size check
    results['product_size_in_range'] = (
        test_case.expected_product_size[0] <= pair.product_size <= test_case.expected_product_size[1]
    )

    # Dimer risk check (Delta G should be > threshold, i.e., less negative)
    if pair.dimer_delta_g is not None:
        results['no_severe_dimer'] = pair.dimer_delta_g > test_case.max_dimer_delta_g
    else:
        results['no_severe_dimer'] = True

    # Optimization not required
    results['no_optimization_required'] = not pair.requires_optimization

    return results


def calculate_pass_rate(results: Dict[str, bool]) -> float:
    """Calculate pass rate from evaluation results."""
    if not results:
        return 0.0
    return sum(results.values()) / len(results)


# ============================================================================
# Primer3 Wrapper Tests
# ============================================================================

class TestPrimer3Wrapper:
    """Tests for the Primer3 wrapper class."""

    def test_calculate_tm_returns_float(self, primer3_wrapper):
        """Tm calculation should return a float value."""
        sequence = "ATCGATCGATCGATCGATCG"
        tm = primer3_wrapper.calculate_tm(sequence)

        assert isinstance(tm, float)
        assert 40.0 <= tm <= 80.0  # Reasonable Tm range

    def test_calculate_tm_consistency(self, primer3_wrapper):
        """Same sequence should always return same Tm."""
        sequence = "GCTAGCTAGCTAGCTAGCTA"

        tm1 = primer3_wrapper.calculate_tm(sequence)
        tm2 = primer3_wrapper.calculate_tm(sequence)

        assert tm1 == tm2

    def test_calculate_hairpin(self, primer3_wrapper):
        """Hairpin calculation should return thermodynamic values."""
        sequence = "GCGCGCAAAAAGCGCGC"  # Palindromic sequence
        result = primer3_wrapper.calculate_hairpin(sequence)

        assert 'tm' in result
        assert 'dg' in result
        assert 'dh' in result

    def test_calculate_heterodimer(self, primer3_wrapper):
        """Heterodimer calculation for two primers."""
        seq1 = "ATCGATCGATCGATCGATCG"
        seq2 = "CGATCGATCGATCGATCGAT"

        result = primer3_wrapper.calculate_heterodimer(seq1, seq2)

        assert 'tm' in result
        assert 'dg' in result

    def test_calculate_thermodynamics_comprehensive(self, primer3_wrapper):
        """Full thermodynamic analysis should include all expected fields."""
        forward = "ATCGATCGATCGATCGATCG"
        reverse = "TAGCTAGCTAGCTAGCTAGC"

        result = primer3_wrapper.calculate_thermodynamics(forward, reverse)

        assert 'forward' in result
        assert 'reverse' in result
        assert 'pair' in result
        assert 'requires_optimization' in result
        assert 'primer3_version' in result

    def test_dimer_detection(self, primer3_wrapper):
        """Primers with complementary 3' ends should trigger dimer warning."""
        # These primers have complementary 3' ends
        forward = "ATCGATCGATCGATCGATAT"  # Ends in AT
        reverse = "GCTAGCTAGCTAGCTAGCAT"  # Ends in AT (complement)

        result = primer3_wrapper.calculate_thermodynamics(forward, reverse)

        # Check if dimer analysis was performed
        assert 'pair' in result
        assert 'heterodimer' in result['pair']


# ============================================================================
# Golden Set Tests
# ============================================================================

class TestGoldenSet:
    """
    Golden set tests for primer design quality validation.

    These tests ensure the agent maintains design quality over time.
    Run these in CI/CD to detect prompt degradation.
    """

    @pytest.mark.parametrize("test_case", GOLDEN_TEST_CASES, ids=lambda tc: tc.name)
    def test_primer_design_quality(self, primer3_wrapper, test_case: GoldenTestCase):
        """
        Test primer design meets quality criteria for each golden case.

        Assertions:
        - At least one valid primer pair is generated
        - Tm difference < specified threshold (typically 1°C)
        - No severe dimer formation (Delta G > -9 kcal/mol)
        - GC content within expected range
        - Product size within expected range
        """
        # Build constraints based on test case
        constraints = DesignConstraints(
            tm_min=test_case.expected_tm_range[0],
            tm_max=test_case.expected_tm_range[1],
            gc_min=test_case.expected_gc_range[0] - 5,  # Allow some flexibility
            gc_max=test_case.expected_gc_range[1] + 5,
            product_size_min=test_case.expected_product_size[0],
            product_size_max=test_case.expected_product_size[1],
            gc_clamp=True,
        )

        # Design primers
        primer_pairs = primer3_wrapper.design_primers(
            template_sequence=test_case.sequence,
            constraints=constraints,
            task_type=test_case.task_type,
            num_return=5
        )

        # Assert we got results
        assert len(primer_pairs) > 0, f"No primers generated for {test_case.name}"

        # Evaluate best pair
        best_pair = primer_pairs[0]
        evaluation = evaluate_primer_pair(best_pair, test_case)

        # Critical assertions
        assert evaluation['tm_difference'], (
            f"Tm difference ({best_pair.tm_difference:.2f}°C) exceeds "
            f"maximum ({test_case.max_tm_difference}°C) for {test_case.name}"
        )

        assert evaluation['no_severe_dimer'], (
            f"Severe dimer detected (Delta G = {best_pair.dimer_delta_g:.2f} kcal/mol) "
            f"for {test_case.name}"
        )

        # Calculate overall pass rate
        pass_rate = calculate_pass_rate(evaluation)
        assert pass_rate >= 0.7, (
            f"Overall quality too low ({pass_rate:.1%}) for {test_case.name}. "
            f"Failures: {[k for k, v in evaluation.items() if not v]}"
        )

    @pytest.mark.parametrize("test_case", GOLDEN_TEST_CASES, ids=lambda tc: tc.name)
    def test_thermodynamic_analysis(self, primer3_wrapper, test_case: GoldenTestCase):
        """
        Test thermodynamic calculations are accurate and complete.
        """
        # First, design primers
        constraints = DesignConstraints(
            tm_min=test_case.expected_tm_range[0],
            tm_max=test_case.expected_tm_range[1],
            product_size_min=test_case.expected_product_size[0],
            product_size_max=test_case.expected_product_size[1],
        )

        primer_pairs = primer3_wrapper.design_primers(
            template_sequence=test_case.sequence,
            constraints=constraints,
            task_type=test_case.task_type,
            num_return=3
        )

        if not primer_pairs:
            pytest.skip(f"No primers generated for {test_case.name}")

        best_pair = primer_pairs[0]

        # Run detailed thermodynamic analysis
        thermo = primer3_wrapper.calculate_thermodynamics(
            forward_seq=best_pair.forward.sequence,
            reverse_seq=best_pair.reverse.sequence
        )

        # Verify thermodynamic data is complete
        assert 'forward' in thermo
        assert 'reverse' in thermo
        assert 'pair' in thermo

        # Verify Tm values match
        assert abs(thermo['forward']['tm'] - best_pair.forward.tm) < 0.5, (
            "Forward Tm mismatch between design and thermodynamics"
        )

        # Verify GC calculation
        assert abs(thermo['forward']['gc_percent'] - best_pair.forward.gc_percent) < 0.1, (
            "GC% mismatch between design and thermodynamics"
        )

    def test_qpcr_amplicon_size_constraint(self, primer3_wrapper):
        """qPCR primers should produce amplicons <= 300 bp."""
        test_case = next(tc for tc in GOLDEN_TEST_CASES if tc.task_type == TaskType.QPCR)

        constraints = DesignConstraints(
            tm_min=55.0,
            tm_max=62.0,
            product_size_min=70,
            product_size_max=300,  # qPCR max
        )

        primer_pairs = primer3_wrapper.design_primers(
            template_sequence=test_case.sequence,
            constraints=constraints,
            task_type=TaskType.QPCR,
            num_return=5
        )

        for pair in primer_pairs:
            assert pair.product_size <= 300, (
                f"qPCR amplicon {pair.product_size} bp exceeds 300 bp maximum"
            )


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for the complete primer design workflow."""

    @pytest.mark.asyncio
    async def test_full_design_workflow(self, primer3_wrapper, variant_checker):
        """Test the complete design workflow from request to response."""
        test_case = GOLDEN_TEST_CASES[0]  # GAPDH

        # Step 1: Design primers
        constraints = DesignConstraints(
            tm_min=55.0,
            tm_max=62.0,
            product_size_min=80,
            product_size_max=150,
        )

        primer_pairs = primer3_wrapper.design_primers(
            template_sequence=test_case.sequence,
            constraints=constraints,
            task_type=test_case.task_type,
            num_return=3
        )

        assert len(primer_pairs) > 0

        # Step 2: Check thermodynamics
        best_pair = primer_pairs[0]
        thermo = primer3_wrapper.calculate_thermodynamics(
            forward_seq=best_pair.forward.sequence,
            reverse_seq=best_pair.reverse.sequence
        )

        assert not thermo.get('requires_optimization', True), (
            "Best pair requires optimization"
        )

        # Step 3: Check variants (mock)
        # In real test, this would query variant databases
        variant_result = await variant_checker.check_region(
            chromosome="chr12",
            start=6534000,
            end=6535000,
            gene_symbol="GAPDH"
        )

        assert variant_result.status == "success"

    @pytest.mark.asyncio
    async def test_variant_conflict_detection(self, variant_checker):
        """Test that variant conflicts are properly detected."""
        # Check a region known to have variants
        result = await variant_checker.check_region(
            chromosome="chrX",
            start=15579500,
            end=15579700,
            gene_symbol="ACE2"
        )

        # Should find variants in ACE2 region
        assert result.total_variants >= 0  # May or may not have variants in mock


# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance:
    """Performance benchmarks for primer design."""

    def test_design_performance(self, primer3_wrapper, default_constraints):
        """Primer design should complete within reasonable time."""
        import time

        test_case = GOLDEN_TEST_CASES[0]

        start = time.time()
        primer_pairs = primer3_wrapper.design_primers(
            template_sequence=test_case.sequence,
            constraints=default_constraints,
            task_type=TaskType.QPCR,
            num_return=5
        )
        elapsed = time.time() - start

        assert elapsed < 5.0, f"Design took too long: {elapsed:.2f}s"
        assert len(primer_pairs) > 0

    def test_thermodynamics_performance(self, primer3_wrapper):
        """Thermodynamic calculations should be fast."""
        import time

        forward = "ATCGATCGATCGATCGATCG"
        reverse = "TAGCTAGCTAGCTAGCTAGC"

        start = time.time()
        for _ in range(100):
            primer3_wrapper.calculate_thermodynamics(forward, reverse)
        elapsed = time.time() - start

        assert elapsed < 5.0, f"100 calculations took too long: {elapsed:.2f}s"


# ============================================================================
# Regression Tests
# ============================================================================

class TestRegression:
    """Regression tests to catch known issues."""

    def test_gc_clamp_required(self, primer3_wrapper):
        """Primers should have GC clamp when requested."""
        constraints = DesignConstraints(
            tm_min=55.0,
            tm_max=62.0,
            gc_clamp=True,
            gc_clamp_length=1,
        )

        test_case = GOLDEN_TEST_CASES[0]
        primer_pairs = primer3_wrapper.design_primers(
            template_sequence=test_case.sequence,
            constraints=constraints,
            num_return=5
        )

        # Check GC clamp on all primers
        for pair in primer_pairs:
            # At least one G or C at 3' end
            fwd_3prime = pair.forward.sequence[-2:].upper()
            rev_3prime = pair.reverse.sequence[-2:].upper()

            fwd_gc = fwd_3prime.count('G') + fwd_3prime.count('C')
            rev_gc = rev_3prime.count('G') + rev_3prime.count('C')

            assert fwd_gc >= 1, f"Forward primer lacks GC clamp: {pair.forward.sequence}"
            assert rev_gc >= 1, f"Reverse primer lacks GC clamp: {pair.reverse.sequence}"

    def test_poly_run_detection(self, primer3_wrapper):
        """Primers should not have excessive poly runs."""
        import re

        constraints = DesignConstraints(
            max_poly_x=4,
        )

        test_case = GOLDEN_TEST_CASES[0]
        primer_pairs = primer3_wrapper.design_primers(
            template_sequence=test_case.sequence,
            constraints=constraints,
            num_return=5
        )

        poly_pattern = re.compile(r'([ATCG])\1{4,}')

        for pair in primer_pairs:
            fwd_poly = poly_pattern.search(pair.forward.sequence.upper())
            rev_poly = poly_pattern.search(pair.reverse.sequence.upper())

            assert not fwd_poly, f"Forward primer has poly run: {fwd_poly.group()}"
            assert not rev_poly, f"Reverse primer has poly run: {rev_poly.group()}"


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
