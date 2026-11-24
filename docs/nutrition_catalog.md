# Curated nutrition catalog

## Provenance and use

The backend seeds a small, fixed catalog from USDA FoodData Central Standard Reference (SR) Legacy. USDA identifies SR Legacy as its final release from April 2018; each row stores the FDC identifier and the catalog version so a reviewer can open the original record. The audit cross-checked every seeded food description and the energy, protein, carbohydrate, and fat entries against the official downloadable SR Legacy CSV archive (`food.csv`, `food_nutrient.csv`, `nutrient.csv`, and `food_portion.csv`) on 2026-11-24. See the [FDC API guide](https://fdc.nal.usda.gov/api-guide/), [FDC data type documentation](https://fdc.nal.usda.gov/data-documentation/), and [download/version history](https://fdc.nal.usda.gov/download-datasets/). The API requires a data.gov key; SmartNutri does not call it at runtime or store an API key.

All nutrition values below are per 100 g: energy in kcal and protein, carbohydrate, and fat in grams. No values are inferred from FoodSeg103. FDC identifiers are source identifiers, not SmartNutri IDs. FoodSeg103 IDs below are class associations only; the segmentation label does not establish cooking state, exact cut, recipe, oil, or ingredients.

| Food identity | Preparation in source record | FoodSeg103 class | kcal | Protein (g) | Carbohydrate (g) | Fat (g) | Source record |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| White long-grain regular enriched rice | Cooked | 66 rice | 130 | 2.69 | 28.17 | 0.28 | [FDC 168878](https://fdc.nal.usda.gov/food-details/168878/nutrients) |
| White long-grain regular enriched rice | Raw | 66 rice | 365 | 7.13 | 79.95 | 0.66 | [FDC 168877](https://fdc.nal.usda.gov/food-details/168877/nutrients) |
| Chicken breast, meat only | Roasted, cooked | 48 chicken duck | 165 | 31.02 | 0 | 3.57 | [FDC 171477](https://fdc.nal.usda.gov/food-details/171477/nutrients) |
| Chicken breast, meat only | Raw, skinless/boneless | 48 chicken duck | 120 | 22.50 | 0 | 2.62 | [FDC 171077](https://fdc.nal.usda.gov/food-details/171077/nutrients) |
| Domesticated duck, meat only | Roasted, cooked | 48 chicken duck | 201 | 23.48 | 0 | 11.20 | [FDC 172411](https://fdc.nal.usda.gov/food-details/172411/nutrients) |
| Fast-food potato, French fried in vegetable oil | French-fried reference | 3 french fries | 312 | 3.43 | 41.44 | 14.73 | [FDC 170698](https://fdc.nal.usda.gov/food-details/170698/nutrients) |
| Golden Delicious apple with skin | Raw | 25 apple | 57 | 0.28 | 13.60 | 0.15 | [FDC 168202](https://fdc.nal.usda.gov/food-details/168202/nutrients) |
| Banana | Raw | 29 banana | 89 | 1.09 | 22.84 | 0.33 | [FDC 173944](https://fdc.nal.usda.gov/food-details/173944/nutrients) |
| Broccoli | Boiled, drained, without salt | 87 broccoli | 35 | 2.38 | 7.18 | 0.41 | [FDC 169967](https://fdc.nal.usda.gov/food-details/169967/nutrients) |
| Broccoli | Raw | 87 broccoli | 34 | 2.82 | 6.64 | 0.37 | [FDC 170379](https://fdc.nal.usda.gov/food-details/170379/nutrients) |
| Red ripe tomatoes | Raw, year-round average | 73 tomato | 18 | 0.88 | 3.89 | 0.20 | [FDC 170457](https://fdc.nal.usda.gov/food-details/170457/nutrients) |

This is 11 source records across eight food identities. The source identifier stored in SQLite is `FDC:<number>` and the version is `SR Legacy final release, April 2018`.

## Safe mapping behavior

- Rice, chicken, and broccoli have multiple source preparations for one FoodSeg103 class. The model cannot select raw versus cooked, so automatic lookup returns no nutrition value until the user chooses a record.
- FoodSeg103 class 48 is literally `chicken duck`. The model cannot determine which animal is present. Chicken and duck remain distinct catalog identities and both require explicit identification/preparation selection.
- Apple's class does not identify variety or skin state. The catalog record is specifically raw Golden Delicious with skin. Other apple forms require an explicit suitable source record.
- Banana and tomato records are raw; they are not reused for cooked or prepared products.
- The fries row is a fast-food French-fried-in-vegetable-oil reference, not a universal value for all fry recipes. Treat the returned value as a database-based estimate and keep it reviewable.
- Mock scan fixtures use identifiers prefixed `test-fixture:` and are isolated from USDA records. A simulated scan cannot resolve to a real USDA record and a real scan cannot resolve to a test fixture.
- No food record is synthesized from a segmentation label. If an identity/preparation is unknown or a source mapping is missing, item nutrients and meal totals remain null until correction.

When a user corrects a meal, the dashboard can pass the selected `nutrition_record_id` for each food item. Nutrition is computed as `nutrient_per_100g Ã— corrected_weight_g / 100`. The backend preserves the measured total food mass separately from estimated or corrected per-food masses and marks calculated nutrient output as estimated.

## Maintenance

The seed is idempotent and rejects conflicting values for an already-seeded `(source, source_identifier)` pair. To update a record, verify its exact FDC food description and nutrients, add a catalog/version change deliberately, and update the catalog tests and this table together. Do not overwrite a preparation-specific item with a generic FoodSeg class average.
