import requests


class Uncertainty(Exception):
    pass


class RefMeta(type):
    """
    Metaclass that implements caching and API-fetching logic for Ref objects.
    """

    def __init__(cls, name, bases, dct):
        """
        Initialize the class. We set up a cache dict here on the class def 
        """
        super().__init__(name, bases, dct)
        cls._cache = {}

    def _get_ref_from_cache(cls, tref: str):
        return cls._cache.get(tref)

    def _set_ref_in_cache(cls, tref: str, oref):
        cls._cache[tref] = oref

    def _has_ref_in_cache(cls, tref: str) -> bool:
        return tref in cls._cache

    def _fetch_ref_from_api(cls, tref: str):
        """
        Fetch data about tref from the Sefaria API. Returns the parsed dict
        (suitable for Ref.__init__), or None if it is not a valid ref.
        """
        url = f"https://www.sefaria.org/api/name/{tref}"
        headers = {"accept": "application/json"}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()

        # If Sefaria says this is not a valid ref, return None
        if not data.get("is_ref"):
            return None

        # Only keep keys that the Ref class is expecting
        ref_data = {key: data[key] for key in cls.keys if key in data}
        return ref_data

    def __call__(cls, tref_or_dict:str | dict, *args, **kwargs):
        """
        Overridden __call__ controls creation of new Ref objects.

        - If `tref_or_dict` is a string, we treat it as a textual ref and check the cache.
        - If tref is not in the cache, fetch from Sefaria, store in the cache,
          and return the newly created def 
        - If tref is already in the cache, return the cached instance.

        If `tref_or_dict` is not a string (e.g., a dict), we just call the base constructor.
        (You can customize this logic to handle more use cases.)
        """
        if isinstance(tref_or_dict, str):
            if cls._has_ref_in_cache(tref_or_dict):
                return cls._get_ref_from_cache(tref_or_dict)
            else:
                ref_data = cls._fetch_ref_from_api(tref_or_dict)
                instance = super().__call__(ref_data, *args, **kwargs)
                cls._set_ref_in_cache(tref_or_dict, instance)
                # If this was loaded with a non-normal form of the ref, store the normal form as well
                if instance.ref != tref_or_dict:
                    cls._set_ref_in_cache(instance.ref, instance)
                return instance
        elif isinstance(tref_or_dict, dict):
            # If `tref_or_dict` is a dict, instantiate directly with class
            return super().__call__(tref_or_dict, *args, **kwargs)
        else:
            raise Exception("Invalid argument type for Ref constructor")


class Ref(metaclass=RefMeta):
    '''
    Represents a reference to a text in Sefaria, with methods for comparison and manipulation.

    Example data, for disambiguation:
        "ref": "Pesach Haggadah, Magid, Yechol Me'rosh Chodesh",
        "url": "Pesach_Haggadah,_Magid,_Yechol_Me'rosh_Chodesh",
        "index": "Pesach Haggadah",
        "book": "Pesach Haggadah, Magid, Yechol Me'rosh Chodesh",

        ----

         "ref": "Shabbat 3b:5",
          "url": "Shabbat.3b.5",
          "index": "Shabbat",
          "book": "Shabbat",
          "internalSections": [
            6,
            5
          ],
          "internalToSections": [
            6,
            5
          ],
          "sections": [
            "3b",
            "5"
          ],
          "toSections": [
            "3b",
            "5"
          ],
    '''

    keys = {
        "is_book",
        "is_node",
        "is_section",
        "is_segment",
        "is_range",
        "ref",
        "url",
        "index",
        "book",
        "internalSections",
        "internalToSections",
        "sections",
        "toSections"
    }

    def __repr__(self):
        return f"<Ref: {getattr(self, 'ref', 'No ref')}>"

    def __init__(self, data: dict):
        data = data or {}
        self.is_book = None
        self.is_node = None
        self.is_section = None
        self.is_segment = None
        self.is_range = None
        self.ref = None
        self.url = None
        self.index = None
        self.book = None
        self.internalSections = []
        self.internalToSections = []
        self.sections = []
        self.toSections = []

        for key, value in data.items():
            setattr(self, key, value)

    def __eq__(self, other):
        if not isinstance(other, Ref):
            return NotImplemented
        if self.book != other.book:
            raise Uncertainty(f"Cannot compare references from different books: {self.book} and {other.book}")
        return self.ref == other.ref

    def __ne__(self, other):
        if not isinstance(other, Ref):
            return NotImplemented
        if self.book != other.book:
            raise Uncertainty(f"Cannot compare references from different books: {self.book} and {other.book}")
        return self.ref != other.ref

    def __lt__(self, other):
        if not isinstance(other, Ref):
            return NotImplemented
        if self.book != other.book:
            raise Uncertainty(f"Cannot compare references from different books: {self.book} and {other.book}")
            
        # Compare starting points first
        my_start = self.starting_ref()
        other_start = other.starting_ref()
        
        # Compare section by section
        for i in range(min(len(my_start.sections), len(other_start.sections))):
            if my_start.sections[i] < other_start.sections[i]:
                return True
            if my_start.sections[i] > other_start.sections[i]:
                return False
                
        # If we get here, all compared sections are equal
        # If one has more sections, it is more specific and should come after
        if len(my_start.sections) < len(other_start.sections):
            return True
        if len(my_start.sections) > len(other_start.sections):
            return False
            
        # Starting points are exactly equal, now compare endpoints
        my_end = self.ending_ref()
        other_end = other.ending_ref()
        
        # If one is a range and one isn't, the range comes after
        if not self.is_range and other.is_range:
            return True
        if self.is_range and not other.is_range:
            return False
            
        # Both are ranges or both are not, compare endpoints
        for i in range(min(len(my_end.sections), len(other_end.sections))):
            if my_end.sections[i] < other_end.sections[i]:
                return True
            if my_end.sections[i] > other_end.sections[i]:
                return False
                
        # Finally, if one has more sections in the endpoint, it comes after
        return len(my_end.sections) < len(other_end.sections)

    def __le__(self, other):
        if not isinstance(other, Ref):
            return NotImplemented
        return self == other or self < other

    def __gt__(self, other):
        if not isinstance(other, Ref):
            return NotImplemented
        return not (self <= other)

    def __ge__(self, other):
        if not isinstance(other, Ref):
            return NotImplemented
        return not (self < other)
    
    def _core_dict(self):
        return {key: getattr(self, key) for key in self.keys}

    @classmethod
    def make_normal(cls, book: str, sections: list[str], to_sections: list[str]) -> str:
        """
        Generate a normal form string from book, sections, and to_sections.
        Note that sections and to_sections are arrays of strings as displayed in the URL, not integers.
        """
        if not sections:
            return book

        i = 0
        for s1, s2 in zip(sections, to_sections):
            if s1 != s2:
                break
            i += 1

        main_portion = ":".join(sections)
        to_portion = ":".join(to_sections[i:])
        return f"{book} {main_portion}{f'-{to_portion}' if to_portion else ''}"

    def normal(self) -> str:
        """Return the normalized string representation of this reference"""
        return self.ref

    def starting_ref(self):
        """
        For ranged Refs, return the starting Ref
        For non-ranged Refs, return self

        :return: :class:`Ref`
        """
        if not self.is_range:
            return self
        d = self._core_dict()
        d["internalToSections"] = self.internalSections[:]
        d["toSections"] = self.sections[:]
        d["is_range"] = False
        d["ref"] = Ref.make_normal(self.book, self.sections, self.sections)
        return Ref(d)

    def ending_ref(self):
        """
        For ranged Refs, return the ending Ref
        For non-ranged Refs, return self

        :return: :class:`Ref`
        """
        if not self.is_range:
            return self
        d = self._core_dict()
        d["internalSections"] = self.internalToSections[:]
        d["sections"] = self.toSections[:]
        d["is_range"] = False
        d["ref"] = Ref.make_normal(self.book, self.toSections, self.toSections)
        return Ref(d)

    def depth(self):
        """Return the number of sections in this reference"""
        return len(self.internalSections)

    def precedes(self, other: "Ref") -> bool:
        """
        Does this Ref completely precede ``other`` Ref?
        True if this ref's endpoint comes before other's starting point.

        :param other: Another reference
        :return bool: True if this reference completely precedes other
        """
        if not isinstance(other, Ref):
            return False
        if not self.book == other.book:
            return False

        my_end = self.ending_ref()
        other_start = other.starting_ref()

        smallest_section_len = min([len(my_end.sections), len(other_start.sections)])

        # Bare book references never precede or follow
        if smallest_section_len == 0:
            return False

        # Compare sections up to the smallest shared depth
        for i in range(smallest_section_len):
            if my_end.sections[i] < other_start.sections[i]:
                return True
            if my_end.sections[i] > other_start.sections[i]:
                return False
                
        # If we've reached here and the sections match exactly, then:
        # If other_start has more sections (is more specific), this doesn't precede it
        # If my_end has more sections (is more specific), this does precede it
        return len(my_end.sections) < len(other_start.sections)

    def follows(self, other: "Ref") -> bool:
        """
        Does this Ref completely follow ``other`` Ref?
        True if this ref's starting point comes after other's endpoint.

        :param other: Another reference
        :return bool: True if this reference completely follows other
        """
        if not isinstance(other, Ref):
            return False
        if not self.book == other.book:
            return False

        my_start = self.starting_ref()
        other_end = other.ending_ref()

        smallest_section_len = min([len(my_start.sections), len(other_end.sections)])

        # Bare book references never precede or follow
        if smallest_section_len == 0:
            return False

        # Compare sections up to the smallest shared depth
        for i in range(smallest_section_len):
            if my_start.sections[i] > other_end.sections[i]:
                return True
            if my_start.sections[i] < other_end.sections[i]:
                return False
                
        # If we've reached here and the sections match exactly, then:
        # If my_start has more sections (is more specific), this follows other
        # If other_end has more sections (is more specific), this doesn't follow it
        return len(my_start.sections) > len(other_end.sections)

    def contains(self, other: "Ref") -> bool:
        """
        Does this Ref completely contain ``other`` Ref?
        
        Tests if this reference encompasses the entirety of the other reference.
        In the case where other is less specific than self, we cannot determine
        containment without additional information.

        :param other: Another reference
        :return bool: True if this reference completely contains other
        :raises Uncertainty: When containment cannot be determined
        """
        if not isinstance(other, Ref):
            return False
            
        # Different books - check if other's book starts with this book
        if not self.book == other.book:
            return other.book.startswith(self.book)

        # If this ref is more specific than other (more sections)
        if len(self.internalSections) > len(other.internalSections):
            additional_depth = len(self.internalSections) - len(other.internalSections)
            # Check if this ref represents a whole unit (like a complete chapter)
            if any([x != 1 for x in self.internalSections[-additional_depth:]]):
                return False  # Not a whole unit, so can't contain a less specific ref
                
            # We don't know the true extent of other, can't determine containment
            raise Uncertainty(f"Cannot determine if {other} is contained in {self}")

        # Get the length of the shorter section list
        smallest_section_len = min([len(self.internalSections), len(other.internalSections)])

        # Check if other's range extends beyond this ref's range
        for i in range(smallest_section_len):
            # If other's end is after this ref's end, this doesn't contain it
            if other.internalToSections[i] > self.internalToSections[i]:
                return False
                
            # If other's end is before this ref's end, no need to check further sections
            if other.internalToSections[i] < self.internalToSections[i]:
                break

        # Check if other's range starts before this ref's range
        for i in range(smallest_section_len):
            # If other's start is before this ref's start, this doesn't contain it
            if other.internalSections[i] < self.internalSections[i]:
                return False
                
            # If other's start is after this ref's start, no need to check further sections
            if other.internalSections[i] > self.internalSections[i]:
                break

        # If we haven't found a reason to return False, this ref contains other
        return True

    def overlaps(self, other: "Ref") -> bool:
        """
        Does this Ref overlap ``other`` Ref?
        
        Tests if there is any intersection between this reference and the other.

        :param other: Another reference
        :return bool: True if the references overlap
        """
        if not isinstance(other, Ref):
            return False
            
        # Different books - check if other's book starts with this book
        if not self.book == other.book:
            return other.book.startswith(self.book)

        # Get the length of the shorter section list
        smallest_section_len = min([len(self.internalSections), len(other.internalSections)])

        # Check if this ref starts after other's end
        for i in range(smallest_section_len):
            # If this ref starts after other's end, there's no overlap
            if self.internalSections[i] > other.internalToSections[i]:
                return False
                
            # If this ref starts before other's end, need to check if other starts before this ref's end
            if self.internalSections[i] < other.internalToSections[i]:
                break

        # Check if this ref ends before other's start
        for i in range(smallest_section_len):
            # If this ref ends before other's start, there's no overlap
            if self.internalToSections[i] < other.internalSections[i]:
                return False
                
            # If this ref ends after other's start, they overlap
            if self.internalToSections[i] > other.internalSections[i]:
                break

        # If we haven't found a reason to return False, the refs overlap
        return True

    def outspecifies(self, other: "Ref") -> bool:
        """
        Is `self` more specific or more comprehensive than `other`?
        
        Determines if this reference should be preferred over the other reference.
        The rules for preference are:
        1. Greater detail (more sections) is preferred over less detail
        2. A reference that contains the other is preferred
        3. A larger range at the same level of detail is preferred
        
        :param other: Another reference to compare with
        :return bool: True if this reference should be preferred over the other
        :raises Uncertainty: When references are from different books
        """
        if not isinstance(other, Ref):
            return False
            
        # Cannot compare references from different books
        if not self.book == other.book:
            raise Uncertainty(f"Can't compare specificity of different books: {self.book} and {other.book}")

        # Rule 1: More detail is preferred
        if len(self.internalSections) > len(other.internalSections):
            return True
            
        # Rule 2: If this ref contains the other, it's preferred
        try:
            if self.contains(other):
                return True
        except Uncertainty:
            # If we can't determine containment, continue with other rules
            pass
            
        # Rule 3: At the same level of detail, larger range is preferred
        if len(self.internalSections) == len(other.internalSections):
            total_self_range = 0
            total_other_range = 0
            
            # Sum up the width of the range at each level
            for i in range(len(self.internalSections)):
                self_range = self.internalToSections[i] - self.internalSections[i]
                other_range = other.internalToSections[i] - other.internalSections[i]
                
                # If ranges differ at this level, the larger one is preferred
                if self_range > other_range:
                    return True
                if self_range < other_range:
                    return False
                    
                total_self_range += self_range
                total_other_range += other_range
                
            # If totals differ, the larger one is preferred
            if total_self_range > total_other_range:
                return True
                
        # This ref doesn't outspecify the other
        return False


class RefSet:
    """
    Store refs in a list and provide methods for manipulation and deduplication.

    RefSet handles collections of references, particularly for the purpose of
    deduplicating and finding the most specific references when there are overlaps.
    """

    def __init__(self, refs: list[Ref]):
        """
        Initialize a RefSet with a list of references.

        :param refs: List of Ref objects
        """
        assert all(isinstance(ref, Ref) for ref in refs)
        self.refs = refs
        # Store a sorted copy for optimized operations
        try:
            self.sorted_refs = sorted(self.refs)
        except Uncertainty:
            # If we get an Uncertainty error during sorting (different books),
            # we'll group by book and sort each group separately
            self.sorted_refs = self._sort_by_book()

    def _sort_by_book(self) -> list[Ref]:
        """
        Sort references by book and then within each book group.
        Used when references from different books can't be directly compared.

        :return: List of sorted references grouped by book
        """
        # Group refs by book
        book_groups = {}
        for ref in self.refs:
            book = ref.book
            if book not in book_groups:
                book_groups[book] = []
            book_groups[book].append(ref)

        # Sort each group and combine
        sorted_refs = []
        for book in sorted(book_groups.keys()):
            try:
                sorted_refs.extend(sorted(book_groups[book]))
            except Uncertainty:
                # If still can't sort, just add them unsorted
                sorted_refs.extend(book_groups[book])

        return sorted_refs

    def __repr__(self):
        return f"<RefSet: {getattr(self, 'refs', 'No refs')}>"

    def __eq__(self, other):
        """
        Check if two RefSets are equal by comparing their sorted references.

        :param other: Another RefSet
        :return: True if the RefSets contain the same references
        """
        if not isinstance(other, RefSet):
            return False
        if len(self.sorted_refs) != len(other.sorted_refs):
            return False

        # Group refs by book for comparison
        self_by_book = self._group_by_book()
        other_by_book = other._group_by_book()

        # Compare books
        if set(self_by_book.keys()) != set(other_by_book.keys()):
            return False

        # Compare refs within each book
        for book in self_by_book:
            self_refs = sorted(self_by_book[book], key=lambda r: r.ref)
            other_refs = sorted(other_by_book[book], key=lambda r: r.ref)

            if len(self_refs) != len(other_refs):
                return False

            for i in range(len(self_refs)):
                if self_refs[i].ref  != other_refs[i].ref:
                    return False

        return True

    def _group_by_book(self) -> dict:
        """
        Group references by book.

        :return: Dictionary mapping book names to lists of references
        """
        result = {}
        for ref in self.refs:
            book = ref.book
            if book not in result:
                result[book] = []
            result[book].append(ref)
        return result

    def as_list(self) -> list[Ref]:
        """
        Return the references as a list.

        :return: List of references
        """
        return self.refs

    def deduplicate(self) -> "RefSet":
        """
        Remove duplicate and redundant references.

        Given a list of refs (some might be segments, sections, or ranges),
        return a list of non-overlapping refs that represent the same content
        in the most specific way possible.

        Rules:
        1. If a ref contains another ref, keep the more specific one
        2. If refs overlap, keep the one that outspecifies the others
        3. Group by book to handle cross-book references

        :return: A new RefSet with deduplicated references
        """
        if not self.refs:
            return RefSet([])

        # Group references by book
        refs_by_book = self._group_by_book()
        deduplicated_refs = []

        # Process each book separately
        for book, book_refs in refs_by_book.items():
            # Skip empty book groups
            if not book_refs:
                continue

            # Sort references within this book
            try:
                sorted_book_refs = sorted(book_refs)
            except Uncertainty:
                # If we can't sort, we'll still try to deduplicate as best we can
                sorted_book_refs = book_refs

            # Find clusters of overlapping references
            clusters = self._find_overlap_clusters(sorted_book_refs)

            # Select the best reference from each cluster
            for cluster in clusters:
                if not cluster:
                    continue

                best_ref = self._find_best_ref(cluster)
                if best_ref:
                    deduplicated_refs.append(best_ref)

        return RefSet(deduplicated_refs)

    def _find_overlap_clusters(self, refs: list[Ref]) -> list[list[Ref]]:
        """
        Find clusters of overlapping references.

        :param refs: List of references to cluster
        :return: List of clusters, where each cluster is a list of overlapping references
        """
        if not refs:
            return []

        # Start with each reference in its own cluster
        clusters = [[ref] for ref in refs]
        
        # Compare each cluster with every other cluster
        i = 0
        while i < len(clusters):
            j = i + 1
            while j < len(clusters):
                # Check if any reference in cluster i overlaps with any reference in cluster j
                merged = False
                for ref_i in clusters[i]:
                    for ref_j in clusters[j]:
                        try:
                            if ref_i.overlaps(ref_j) or ref_i.contains(ref_j) or ref_j.contains(ref_i):
                                # Merge clusters and break
                                clusters[i].extend(clusters[j])
                                clusters.pop(j)
                                merged = True
                                break
                        except Uncertainty:
                            # If we can't determine overlap, keep them separate
                            pass
                    if merged:
                        break
                
                if not merged:
                    j += 1
            i += 1

        return clusters

    def _find_best_ref(self, refs: list[Ref]) -> Ref:
        """
        Find the best (most specific) reference from a list of overlapping references.

        :param refs: List of references to compare
        :return: The most specific reference
        """
        if not refs:
            return None
        if len(refs) == 1:
            return refs[0]

        # Start with the first reference as the best
        best = refs[0]

        # Compare each reference against the current best
        for candidate in refs[1:]:
            try:
                if candidate.outspecifies(best):
                    best = candidate
            except Uncertainty:
                # If we can't compare, keep the current best
                continue

        return best