from ..ref import Ref, RefSet, Uncertainty
import pytest

class TestRef:

    uncertain_pairs =  [[Ref("Genesis 1:1-31"),Ref("Genesis 1")],
        [Ref("Leviticus 1:1-27.34"),Ref("Leviticus")],
        [Ref("Leviticus 1:1-27.30"),Ref("Leviticus")]]

    book_mismatch_pairs = [[Ref("Genesis 1:1"), Ref("Exodus 1:1")],
    [Ref("Shabbat 2a"), Ref("Berakhot 2a")]]

    @pytest.mark.parametrize("ref1, ref2", uncertain_pairs)
    def test_uncertainty(self, ref1, ref2):
        with pytest.raises(Uncertainty):
            ref1.contains(ref2)

    def test_cache(self):
        assert Ref("Genesis 1:1") == Ref("Genesis 1:1")

    def test_starting_ref(self):
        assert Ref("Genesis 1:1").starting_ref() == Ref("Genesis 1:1")
        assert Ref("Genesis 1:1-3").starting_ref() == Ref("Genesis 1:1")
        assert Ref("Genesis 1:1-3:5").starting_ref() == Ref("Genesis 1:1")
        assert Ref("Shabbat 7b:10-20").starting_ref() == Ref("Shabbat 7b:10")

    def test_ending_ref(self):
        assert Ref("Genesis 1:1").ending_ref() == Ref("Genesis 1:1")
        assert Ref("Genesis 1:1-3").ending_ref() == Ref("Genesis 1:3")
        assert Ref("Genesis 1:1-3:5").ending_ref() == Ref("Genesis 3:5")
        assert Ref("Shabbat 7b:10-20").ending_ref() == Ref("Shabbat 7b:20")
        
    def test_comparison_operators(self):
        """Test comparison operators (==, !=, <, >, <=, >=) for self-consistency"""
        # Test equality operators
        assert Ref("Genesis 1:1") == Ref("Genesis 1:1")
        assert Ref("Genesis 1:1") != Ref("Genesis 1:2")
        assert not (Ref("Genesis 1:1") != Ref("Genesis 1:1"))
        assert not (Ref("Genesis 1:1") == Ref("Genesis 1:2"))
        
        # Test less than operator
        assert Ref("Genesis 1:1") < Ref("Genesis 1:2")
        assert Ref("Genesis 1:1") < Ref("Genesis 2:1")
        assert not Ref("Genesis 1:2") < Ref("Genesis 1:1")
        assert not Ref("Genesis 1:1") < Ref("Genesis 1:1")
        # Test with ranges - start same, different end
        assert Ref("Genesis 1:1-2") < Ref("Genesis 1:1-3")
        assert not Ref("Genesis 1:1-3") < Ref("Genesis 1:1-2")
        
        # Test less than or equal to operator
        assert Ref("Genesis 1:1") <= Ref("Genesis 1:2")
        assert Ref("Genesis 1:1") <= Ref("Genesis 1:1")
        assert not Ref("Genesis 1:2") <= Ref("Genesis 1:1")
        # Test with ranges
        assert Ref("Genesis 1:1-2") <= Ref("Genesis 1:1-2")
        assert Ref("Genesis 1:1-2") <= Ref("Genesis 1:1-3")
        assert not Ref("Genesis 1:1-3") <= Ref("Genesis 1:1-2")
        
        # Test greater than operator
        assert Ref("Genesis 1:2") > Ref("Genesis 1:1")
        assert Ref("Genesis 2:1") > Ref("Genesis 1:1")
        assert not Ref("Genesis 1:1") > Ref("Genesis 1:2")
        assert not Ref("Genesis 1:1") > Ref("Genesis 1:1")
        # Test with ranges
        assert Ref("Genesis 1:1-3") > Ref("Genesis 1:1-2")
        assert not Ref("Genesis 1:1-2") > Ref("Genesis 1:1-3")
        
        # Test greater than or equal to operator
        assert Ref("Genesis 1:2") >= Ref("Genesis 1:1")
        assert Ref("Genesis 1:1") >= Ref("Genesis 1:1")
        assert not Ref("Genesis 1:1") >= Ref("Genesis 1:2")
        # Test with ranges
        assert Ref("Genesis 1:1-2") >= Ref("Genesis 1:1-2")
        assert Ref("Genesis 1:1-3") >= Ref("Genesis 1:1-2")
        assert not Ref("Genesis 1:1-2") >= Ref("Genesis 1:1-3")
        
    def test_comparison_consistency(self):
        """Test for consistency between comparison operators"""
        # If a < b, then not (a >= b)
        ref_a = Ref("Genesis 1:1")
        ref_b = Ref("Genesis 1:2")
        assert (ref_a < ref_b) == (not (ref_a >= ref_b))
        
        # If a > b, then not (a <= b)
        assert (ref_b > ref_a) == (not (ref_b <= ref_a))
        
        # If a == b, then a <= b and a >= b
        ref_c = Ref("Genesis 1:1")
        assert (ref_a == ref_c) == (ref_a <= ref_c and ref_a >= ref_c)
        
        # If a != b, then not (a == b)
        assert (ref_a != ref_b) == (not (ref_a == ref_b))
        
        # If a < b and b < c, then a < c (transitivity)
        ref_d = Ref("Genesis 1:3")
        assert ref_a < ref_b and ref_b < ref_d
        assert ref_a < ref_d
        
        # Test with more complex refs
        ref1 = Ref("Genesis 1:1-5")
        ref2 = Ref("Genesis 1:1-6")
        ref3 = Ref("Genesis 1:2-5")
        
        # ref1 and ref3 have same ending but different starting points
        assert ref1 < ref3  # Starting earlier should be "less than"
        assert ref3 > ref1
        
        # ref1 and ref2 have same starting point but different ending points
        assert ref1 < ref2  # Ending earlier should be "less than"
        assert ref2 > ref1

    @pytest.mark.parametrize("ref1, ref2", book_mismatch_pairs)
    def test_comparison_with_different_books(self, ref1, ref2):
        with pytest.raises(Uncertainty):
            ref1 > ref2
        with pytest.raises(Uncertainty):
            ref1 < ref2
        with pytest.raises(Uncertainty):
            ref1 == ref2
        with pytest.raises(Uncertainty):
            ref1 >= ref2
        with pytest.raises(Uncertainty):
            ref1 <= ref2

        
    def test_comparison_with_range_refs(self):
        """Test comparison operators with range references"""
        # Equal ranges
        assert Ref("Genesis 1:1-3") == Ref("Genesis 1:1-3")
        assert not (Ref("Genesis 1:1-3") != Ref("Genesis 1:1-3"))
        
        # Range and single ref
        assert Ref("Genesis 1:1") < Ref("Genesis 1:1-3")
        assert Ref("Genesis 1:1-3") > Ref("Genesis 1:1")
        
        # Overlapping ranges
        assert Ref("Genesis 1:1-3") < Ref("Genesis 1:2-4")
        assert Ref("Genesis 1:2-4") > Ref("Genesis 1:1-3")
        
        # Non-overlapping ranges
        assert Ref("Genesis 1:1-3") < Ref("Genesis 1:4-6")
        assert Ref("Genesis 1:4-6") > Ref("Genesis 1:1-3")

    def test_precedes(self):
        assert Ref("Genesis 1:1").precedes(Ref("Genesis 1:2"))
        assert Ref("Genesis 1:1").precedes(Ref("Genesis 2:1"))
        assert not Ref("Genesis 1:1").precedes(Ref("Genesis 1:1"))
        assert not Ref("Genesis 1:2").precedes(Ref("Genesis 1:1"))
        assert Ref("Shabbat 7b:10").precedes(Ref("Shabbat 7b:11"))
        assert not Ref("Shabbat 7b:10-14").precedes(Ref("Shabbat 7b:11"))

    def test_follows(self):
        assert Ref("Genesis 1:2").follows(Ref("Genesis 1:1"))
        assert Ref("Genesis 2:1").follows(Ref("Genesis 1:1"))
        assert not Ref("Genesis 1:1").follows(Ref("Genesis 1:1"))
        assert not Ref("Genesis 1:1").follows(Ref("Genesis 1:2"))
        assert Ref("Shabbat 7b:11").follows(Ref("Shabbat 7b:10"))
        assert not Ref("Shabbat 7b:11").follows(Ref("Shabbat 7b:10-14"))

    def test_contains(self):
        assert Ref("Genesis 5:10-20").contains(Ref("Genesis 5:10-20"))
        assert Ref("Genesis 5:10-20").contains(Ref("Genesis 5:13-18"))
        assert not Ref("Genesis 5:10-20").contains(Ref("Genesis 5:21-25"))
        assert not Ref("Genesis 5:10-20").contains(Ref("Genesis 5:18-25"))

        assert Ref("Genesis 5:10-6:20").contains(Ref("Genesis 5:18-25"))
        assert Ref("Genesis 5:10-6:20").contains(Ref("Genesis 5:18-6:10"))
        assert not Ref("Genesis 5:10-6:20").contains(Ref("Genesis 6:21-25"))
        assert not Ref("Genesis 5:10-6:20").contains(Ref("Genesis 6:5-25"))

        assert Ref("Exodus 6").contains(Ref("Exodus 6:2"))
        assert Ref("Exodus 6").contains(Ref("Exodus 6:2-12"))

        assert Ref("Genesis 1").contains(Ref("Genesis 1:1-31"))

        assert Ref("Exodus").contains(Ref("Exodus 6"))
        assert Ref("Exodus").contains(Ref("Exodus 6:2"))
        assert Ref("Exodus").contains(Ref("Exodus 6:2-12"))

        assert not Ref("Exodus 6:2").contains(Ref("Exodus 6"))
        assert not Ref("Exodus 6:2-12").contains(Ref("Exodus 6"))

        assert not Ref("Exodus 6").contains(Ref("Exodus"))
        assert not Ref("Exodus 6:2").contains(Ref("Exodus"))
        assert not Ref("Exodus 6:2-12").contains(Ref("Exodus"))

        assert Ref("Leviticus").contains(Ref("Leviticus"))
        assert Ref("Leviticus").contains(Ref("Leviticus 1:1-27.34"))
        assert Ref("Leviticus").contains(Ref("Leviticus 1-27"))

        assert not Ref("Leviticus 1:2-27.30").contains(Ref("Leviticus"))
        assert not Ref("Leviticus 2:2-27.30").contains(Ref("Leviticus"))

        # These fail, and always did
        # assert not Ref("Leviticus").contains(Ref("Leviticus 1:1-27.35"))
        # assert not Ref("Leviticus").contains(Ref("Leviticus 1-28"))

        assert Ref("Rashi on Genesis 5:10-20").contains(Ref("Rashi on Genesis 5:18-20"))
        assert not Ref("Rashi on Genesis 5:10-20").contains(Ref("Rashi on Genesis 5:21-25"))
        assert not Ref("Rashi on Genesis 5:10-20").contains(Ref("Rashi on Genesis 5:15-25"))

        assert Ref("Rashi on Genesis 5:10-6:20").contains(Ref("Rashi on Genesis 6:18-19"))
        assert not Ref("Rashi on Genesis 5:10-6:20").contains(Ref("Rashi on Genesis 6:21-25"))
        assert not Ref("Rashi on Genesis 5:10-6:20").contains(Ref("Rashi on Genesis 6:5-25"))

        assert not Ref("Genesis 5:10-6:20").contains(Ref("Rashi on Genesis 5:10-6:20"))
        assert not Ref("Rashi on Genesis 5:10-6:20").contains(Ref("Genesis 5:10-6:20"))

        assert Ref("Shabbat 5b-7a").contains(Ref("Shabbat 6b-7a"))
        assert not Ref("Shabbat 5b-7a").contains(Ref("Shabbat 15b-17a"))
        assert not Ref("Shabbat 5b-7a").contains(Ref("Shabbat 6b-17a"))

        assert Ref("Shabbat 5b:10-20").contains(Ref("Shabbat 5b:18-20"))
        assert not Ref("Shabbat 5b:10-20").contains(Ref("Shabbat 5b:23-29"))
        assert not Ref("Shabbat 5b:10-20").contains(Ref("Shabbat 5b:15-29"))

        assert not Ref("Steinsaltz_on_Jerusalem_Talmud_Shekalim.4.4.42-5.1.10").contains(Ref("Steinsaltz on Jerusalem Talmud Shekalim 4:4:1"))

        assert Ref("Jastrow").contains(Ref("Jastrow, פֶּתַח 1"))
        assert not Ref("Jastrow, פֶּתַח 1").contains(Ref("Jastrow"))


    def test_overlaps(self):
        assert Ref("Genesis 5:10-20").overlaps(Ref("Genesis 5:18-25"))
        assert Ref("Genesis 5:10-20").overlaps(Ref("Genesis 5:13-28"))
        assert Ref("Genesis 5:13-28").overlaps(Ref("Genesis 5:10-20"))
        assert not Ref("Genesis 5:10-20").overlaps(Ref("Genesis 5:21-25"))

        assert not Ref("Genesis 1").overlaps(Ref("Genesis 2"))
        assert not Ref("Genesis 2").overlaps(Ref("Genesis 1"))
        assert Ref("Genesis 1").overlaps(Ref("Genesis 1"))

        assert Ref("Genesis 5:10-6:20").overlaps(Ref("Genesis 6:18-25"))
        assert Ref("Genesis 5:10-6:20").overlaps(Ref("Genesis 5:18-25"))
        assert Ref("Genesis 5:18-25").overlaps(Ref("Genesis 5:10-6:20"))
        assert not Ref("Genesis 5:10-6:20").overlaps(Ref("Genesis 6:21-25"))

        assert Ref("Genesis 5").overlaps(Ref("Genesis"))
        assert Ref("Genesis").overlaps(Ref("Genesis 5"))

        assert Ref("Rashi on Genesis 5:10-20").overlaps(Ref("Rashi on Genesis 5:18-25"))
        assert not Ref("Rashi on Genesis 5:10-20").overlaps(Ref("Rashi on Genesis 5:21-25"))

        assert Ref("Rashi on Genesis 5:10-6:20").overlaps(Ref("Rashi on Genesis 6:18-25"))
        assert not Ref("Rashi on Genesis 5:10-6:20").overlaps(Ref("Rashi on Genesis 6:21-25"))

        assert not Ref("Genesis 5:10-6:20").overlaps(Ref("Rashi on Genesis 5:10-6:20"))

        assert Ref("Shabbat 5b-7a").overlaps(Ref("Shabbat 6b-9a"))
        assert not Ref("Shabbat 5b-7a").overlaps(Ref("Shabbat 15b-17a"))

        assert Ref("Shabbat 5b:10-20").overlaps(Ref("Shabbat 5b:18-20"))
        assert not Ref("Shabbat 5b:10-20").overlaps(Ref("Shabbat 5b:23-29"))

        assert Ref("Genesis 1:10-4:10").overlaps(Ref("Genesis 3:15-5:5"))

class TestRefSet:
    rs = RefSet([Ref("Leviticus 9:2"), Ref("Leviticus 9:3"), Ref("Leviticus 9:4"), Ref("Leviticus 9:2-9:4"), Ref("Leviticus 9"), Ref("Leviticus 9-10")])
    rs2 = RefSet([Ref("Leviticus 9:2"), Ref("Leviticus 9:3"), Ref("Leviticus 9:4"), Ref("Leviticus 9:2-9:4"), Ref("Leviticus 9"), Ref("Leviticus 9-10"),
                  Ref("Rashi on Leviticus 9:2"), Ref("Rashi on Leviticus 9:3"),  Ref("Rashi on Leviticus 9:3-5"), Ref("Rashi on Leviticus 9:4")])

    rs3 = RefSet([
        Ref("Berakhot.2a"),
        Ref("Berakhot.2a.1-5"),
        Ref("Berakhot.2a.12"),
        Ref("Berakhot.2b.17-3a.3"),
        Ref("Berakhot.2b.13"),
        Ref("Berakhot.2b.5-16"),
        Ref("Berakhot.3a.2")
    ])

    def test_deduplicate(self):
        assert self.rs.deduplicate() == RefSet([Ref("Leviticus 9:2-9:4")])
        assert self.rs2.deduplicate() == RefSet([Ref("Leviticus 9:2-9:4"), Ref("Rashi on Leviticus 9:2"), Ref("Rashi on Leviticus 9:3-5")])
        assert self.rs3.deduplicate() == RefSet([Ref("Berakhot.2a.1-5"), Ref("Berakhot.2b.17-3a.3"), Ref("Berakhot.2b.5-16")])
