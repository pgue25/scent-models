# Smell survey v2: draft for the pilot

Goal: questions that **split people apart**. v1 asked 9 smells, and several of them probably
measured the same thing (sweet smells together, fresh smells together) while some tastes were
not measured at all. This draft asks more smells in the pilot than the final quiz will keep.
The pilot data then tells us which ones to keep (see "How we pick the final 10").

Wording for every 1 to 6 item (no neutral middle, same as v1):
> "Imagine a perfume, cologne or candle that smells like this. How much would you like it?"
> 1 = Hate it ... 6 = Love it. There is also a "Don't know this smell" box, so a guess is not
> recorded as a real answer.

## Section 1. Everyday smells, pilot pool (20 items, 1 to 6)

Grouped by the taste direction each is meant to measure. Two or three per direction, because
one item is noisy and two are reliable. Items marked ★ are the ones that people tend to split
on: some love them and some hate them. Those split people the most.

| direction | items |
|---|---|
| Fresh / clean | Ocean breeze · Clean laundry · Fresh mint |
| Green / outdoors | Fresh-cut grass · Rain on warm pavement · Pine forest |
| Citrus | Orange peel · Lemon zest |
| Floral | Fresh roses · Lavender ★ · Jasmine / white flowers ★ |
| Fruity / candy | Ripe peaches · Strawberry candy ★ |
| Sweet / gourmand | Vanilla · Caramel / toffee · Coffee beans |
| Warm spice | Cinnamon & cloves · Freshly sliced ginger |
| Smoky / dark | Campfire smoke ★ · Incense / patchouli ★ |
| Powdery / soft | Baby powder ★ |

Twenty one items are listed. Drop one from Fresh or Green before sending if 20 is a hard cap.
v1's Christmas tree and summer rain are kept in spirit (Pine forest, Rain on warm pavement),
reworded to be more specific. "Vanilla candle" becomes "Vanilla" so it isn't about candles.

## Section 2. Forced choice (pick one), 5 pairs

Pairs remove the "I like everything" effect: someone who rates every smell 5 still has to
choose. Each pair crosses two directions we expect to be opposites.

1. Ocean breeze **or** Vanilla
2. Fresh roses **or** Cedar wood
3. Lemon zest **or** Caramel
4. Campfire smoke **or** Clean laundry
5. Ripe peaches **or** Black pepper

## Section 3. Two style questions (not smells, but they divide people)

1. "I like a scent that is..." light and close to the skin / noticeable across the room
2. "I mostly buy scent for..." myself to wear / my home / both / neither

## Section 4. Products you have smelled (the answer key, 1 to 5, optional rows)

Keep v1's grid of 16 products (8 fragrances, 8 candles). These are what the model predicts:
in testing, the model sees a person's quiz answers and predicts the products we hide from it.
Ask people to rate at least 5 products if they can; people with fewer than 4 can still help
the quiz analysis but cannot be scored.

Pushback: 16 specific products is the weak point. Most people have smelled 3 or 4 of them,
which leaves very few hidden products to test on. Consider swapping a few niche ones for very
common ones (a Bath & Body Works best seller, a Yankee Candle classic, Axe / Old Spice).
The team agreed on 2026-11-24 to keep the v1 survey and product notes as they are. This v2
draft is Paul's; check with the team before it replaces the group survey.

## Section 5. Optional

- Products you already love (free text, up to 3), as in v1.
- Age band and the fragrance section you usually shop (women's / men's / both / none).
  Both predict taste strongly and cost one click each.

## How we pick the final 10 (after about 30 pilot responses)

For each everyday smell, compute:
1. **Spread across people.** Low spread (everyone loves clean laundry) means low information. Drop.
2. **Overlap with other items.** If two items correlate above about 0.7, keep the one with the
   bigger spread.
3. **Directions.** Factor analysis of the answers shows how many real taste directions there are.
   Keep about 2 items per real direction.
4. **Usefulness.** How much worse product predictions get when the item is removed.
5. **"Don't know" rate.** Over 20% means too many people are guessing. Drop or reword.

Then rerun the clustering on the pilot with the final items and check whether clearer groups
appear (silhouette and stability, as in CLAUDE.md section 5a). The code for this goes in
`quiz/` in the v2 repo.

Length check: 21 + 5 + 2 + grid + optional is about 4 to 5 minutes on a phone. The final quiz
in the app is about 10 smells plus the 5 pairs, about 1 minute.
