"""Column catalog for the Canadian BDM lettuce survey.

The persona is demographics and shopping habits. Willingness-to-pay bids are
outcomes and are not listed here.
"""

from __future__ import annotations

# Source column, canonical key, label shown on the persona card.
SURVEY_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("survey_q01_age", "age", "Age"),
    ("survey_q02_sex", "sex", "Sex"),
    ("survey_q03_price", "price_importance", "Importance of price when buying produce"),
    ("survey_q03_locally_grown", "locally_grown_importance", "Importance of locally grown produce"),
    ("survey_q03_pesticide_free", "pesticide_free_importance", "Importance of pesticide-free produce"),
    ("survey_q03_brand_name", "brand_importance", "Importance of brand name"),
    ("survey_q04_heard_of_vertical_farming", "heard_of_vertical_farming", "Heard of vertical farming"),
    ("survey_q05_heard_of_greenhouse_growing", "heard_of_greenhouse_growing", "Heard of greenhouse growing"),
    ("survey_q06_primary_grocery", "primary_grocery_shopper", "Primary grocery shopper"),
    ("survey_q07_primary_cook", "primary_cook", "Primary cook"),
    ("survey_q08_leafy_green_meals", "leafy_green_meals", "Leafy-green meals (survey category)"),
    ("survey_q09_not_prepared_meals", "meals_not_prepared_at_home", "Meals not prepared at home (survey category)"),
    ("survey_q10_try_novel_technology", "try_novel_food_technology", "Likelihood of trying food from novel technology"),
    ("survey_q11_grow_own_food", "grows_own_food", "Grows some of their own food"),
    ("survey_q12_variety_concern", "food_variety_concern", "Concern about food variety"),
    ("survey_q13_lack_of_resources", "food_resource_constraints", "Food-resource constraints reported in the survey"),
    ("survey_q14_political", "political", "Political leaning"),
    ("survey_q15_income", "income", "Household income"),
    ("survey_q16_education", "education", "Education"),
    ("survey_q17_household_size", "household_size", "Household size"),
)

BACKGROUND_KEYS = tuple(key for _column, key, _label in SURVEY_FIELDS)

# Outcome column, short product name used in the question stem.
WTP_PRODUCTS: tuple[tuple[str, str], ...] = (
    ("local_field", "local field-grown lettuce"),
    ("local_greenhouse", "local greenhouse-grown lettuce"),
    ("local_vertical", "local vertically farmed lettuce"),
    ("nonlocal_field", "non-local field-grown lettuce"),
    ("nonlocal_greenhouse", "non-local greenhouse-grown lettuce"),
    ("nonlocal_vertical", "non-local vertically farmed lettuce"),
)

WTP_COLUMNS = tuple(column for column, _name in WTP_PRODUCTS)
OUTCOME_COLUMNS = WTP_COLUMNS + ("conclusion_purchased_product",)

PERSONA_PREFACE = (
    "Structured shopper profile from the Canadian BDM lettuce survey "
    "(Gibson and Li). These are demographics and consumer habits, "
    "not a long narrative persona."
)


def wtp_prompt(product: str) -> str:
    return (
        "You are taking part in a Becker-DeGroot-Marschak elicitation in Ontario, Canada. "
        f"What is the most you would pay, in Canadian dollars from 0 to 10, for {product}?"
    )


def purchase_prompt(selected_product: str, drawn_price: str) -> str:
    product = selected_product.strip() or "the randomly selected lettuce"
    price = drawn_price.strip() or "the randomly drawn price"
    return (
        "Under the same Becker-DeGroot-Marschak rule, one product was drawn at random "
        f"({product}) and a price was drawn at random ({price} Canadian dollars). "
        "Would you purchase that product at the drawn price? Answer TRUE or FALSE."
    )


def render_background(demographics: dict[str, str]) -> str:
    """Persona text from background keys only. Outcome columns are ignored."""
    labels = {key: label for _column, key, label in SURVEY_FIELDS}
    lines = [PERSONA_PREFACE, ""]
    for key in BACKGROUND_KEYS:
        value = str(demographics.get(key) or "").strip()
        if not value or key in WTP_COLUMNS:
            continue
        lines.append(f"{labels[key]}: {value}")
    body = "\n".join(lines).strip()
    if body == PERSONA_PREFACE:
        return ""
    return body
