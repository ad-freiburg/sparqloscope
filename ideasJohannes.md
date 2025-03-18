
* Joins between arbitrary predicates might be accidentally empty. We need
  something more involved, probably the complete statistics of all joins of two
  predicates.

* We already had star joins and chain joins etc.

* For GroupBy we have the sort-group algorithm and then the hash-map algorithm.
  We should have 
  * Inputs that can be presorted
  * Inputs that can't be presorted (e.g. a join on the wrong column as input).
  * all of the above with results that have a large number of groups and a small
    number of groups.

* Do we do union graph only, or also FROM/FROM WHERE etc?

* We need UNION + Sort/Join 

* for OPTIONAL we need something like
  * Payload only, inner part of OPTIONAL is simple
  * Payload only, body of OPTIONAL is complex (large joins etc)
  * Chained OPTIONAL as fallback (e.g. labels in different languages)
  

* We need measurements of the difference of certain queries (e.g. Query
  including export - query only count = time for export.
  * Same for expressions (string expressions, numeric expressions, date
    expressions, LANG expressions).

* FILTER:
  * Filters that can be binary search, and filters that can't
  * Filters that can use the realhannes prefiltering.
  * Filters with large results and with small results with similarly complex
    expressions (to learn something about the time it takes to write a result).

* MINUS / EXISTS
  * Todo... similar cases as for join.


* For statistics we also can include more complex statistics
  * (e.g. all joins of predicates + GROUP BY, types of obejcts of predicates,...)
  * Multiplicity queries (number of distinct subjects per Predicate etc.)

* Query planner tests: Queries that are much cheaper if they are rewritten more
  or less complex.
  * Pushing VALUES etc. into UNIONs
  * How are Groups `{}` with and without filters optimized or not 
  * What about subqueries.


  
  * Idea for the writeup:
    * This is a benchmark from the perspective of SPARQL engine implementers, it
      should expose typical optimizations and algorithms to figure out, what
      other engines do. This goes well with the feature based (vs. data-based)
      queries.
    * Do we propose the full ultimate benchmark, or an extensible framework that
      should then be extended by owners of datasets + implementers of tools
