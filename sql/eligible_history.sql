-- Current announcement fields cannot access a same-day or later auction result.
WITH ranked AS (
  SELECT current.key AS key, previous.key AS source_key,
         previous.auction_date AS source_auction_date,
         previous.target AS source_target,
         previous.offering_billions AS source_offering_billions,
         ROW_NUMBER() OVER (
           PARTITION BY current.key
           ORDER BY previous.auction_date DESC, previous.key DESC
         ) AS reverse_position
  FROM auctions AS current
  JOIN scored ON scored.key = current.key
  JOIN auctions AS previous
    ON previous.term_years = current.term_years
   AND previous.auction_date < current.announcement_date
)
SELECT * FROM ranked WHERE reverse_position <= 6
ORDER BY key, reverse_position DESC;
