"""
Food Recommendation Engine
Provides nutrient-rich foods based on the child's nutritional condition.
"""


def generate_food_recommendation(
    underweight,
    stunting,
    wasting,
    age_months
):

    foods = {}

    # -----------------------------
    # Protein
    # -----------------------------
    if underweight or wasting:

        foods["Protein Rich Foods"] = [
            "Eggs",
            "Milk",
            "Curd",
            "Paneer",
            "Fish",
            "Chicken",
            "Soybean",
            "Lentils",
            "Peanuts"
        ]

    # -----------------------------
    # Iron
    # -----------------------------
    if underweight:

        foods["Iron Rich Foods"] = [
            "Spinach",
            "Beans",
            "Dates",
            "Raisins",
            "Jaggery",
            "Liver",
            "Green leafy vegetables"
        ]

    # -----------------------------
    # Calcium
    # -----------------------------
    if stunting:

        foods["Calcium Rich Foods"] = [
            "Milk",
            "Curd",
            "Paneer",
            "Cheese",
            "Ragi",
            "Sesame Seeds"
        ]

    # -----------------------------
    # Vitamin A
    # -----------------------------
    if stunting:

        foods["Vitamin A Rich Foods"] = [
            "Carrot",
            "Pumpkin",
            "Sweet Potato",
            "Mango",
            "Papaya"
        ]

    # -----------------------------
    # Zinc
    # -----------------------------
    if stunting or wasting:

        foods["Zinc Rich Foods"] = [
            "Eggs",
            "Beans",
            "Nuts",
            "Seeds",
            "Whole grains"
        ]

    # -----------------------------
    # Energy Foods
    # -----------------------------
    if wasting:

        foods["Energy Dense Foods"] = [
            "Banana",
            "Peanut Butter",
            "Potato",
            "Rice",
            "Khichdi",
            "Healthy Oils"
        ]

    # -----------------------------
    # Healthy Child
    # -----------------------------
    if len(foods) == 0:

        foods["Balanced Diet"] = [
            "Milk",
            "Eggs",
            "Fruits",
            "Vegetables",
            "Whole grains",
            "Dal",
            "Rice",
            "Seasonal fruits"
        ]

    # -----------------------------
    # Age Advice
    # -----------------------------
    if age_months < 6:

        foods["Age Advice"] = [
            "Exclusive breastfeeding"
        ]

    elif age_months < 24:

        foods["Age Advice"] = [
            "Breastfeeding",
            "Soft complementary foods",
            "Mashed vegetables",
            "Fruit puree"
        ]

    else:

        foods["Age Advice"] = [
            "Three balanced meals",
            "Two healthy snacks",
            "Plenty of water"
        ]

    return foods