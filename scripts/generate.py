#!/usr/bin/env python3
"""Deterministic synthetic retail warehouse generator.

Writes normalized CSV files with stable foreign keys and realistic relationships.
The generator is designed to scale to 25M+ rows without holding the full warehouse
in memory.

This script ONLY generates the raw synthetic CSVs (products, orders, reviews,
etc.). For LLM-based enrichment of products.csv descriptions or reviews.csv
review text, run enrich_descriptions.py / enrich_reviews.py afterward — those
are separate scripts (in llm_common.py + enrich_descriptions.py +
enrich_reviews.py) so this script has no LLM/network dependencies at all.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

SEED = 42


SEED = 42

CATEGORIES = [
    "Outdoor",
    "Clothing",
    "Electronics",
    "Home",
    "Fitness",
    "Kids",
    "Beauty",
    "Sports",
    "Grocery",
    "Toys",
    "Office",
    "Pets",
    "Automotive",
    "Garden",
    "Health",
    "Books",
    "Footwear",
    "Accessories",
    "Travel",
    "Gaming",
    "Appliances",
    "Bedding",
    "Kitchen",
    "Decor",
    "Lighting",
    "Storage",
    "Cleaning",
    "Stationery",
    "Baby",
    "Luggage",
    "Audio",
    "Wearables",
    "Smart Home",
    "Cycling",
    "Running",
    "Camping",
    "Hiking",
    "Fishing",
    "Swimming",
    "Yoga",
    "Skincare",
    "Haircare",
    "Supplements",
    "Gadgets",
    "Computing",
    "Networking",
    "Monitors",
    "Peripherals",
    "Fragrance",
    "Jewelry",
]

# Real, category-appropriate brands. Lists are NOT capped to a fixed size —
# some categories genuinely have more well-known brands than others, and a
# brand can legitimately appear under more than one category (e.g. Nike
# shows up under Clothing, Fitness, Sports, Footwear, and Running).
CATEGORY_BRANDS: Dict[str, List[str]] = {
    "Outdoor": ["REI Co-op", "Patagonia", "The North Face", "Columbia", "Coleman", "YETI", "Osprey", "Black Diamond", "Marmot", "Kelty", "Wildcraft", "Decathlon", "Quechua"],
    "Clothing": ["Nike", "Adidas", "Levi's", "Gap", "H&M", "Zara", "Uniqlo", "Ralph Lauren", "Tommy Hilfiger", "Calvin Klein", "Fabindia", "Allen Solly", "Van Heusen", "Peter England", "Biba", "Raymond", "W for Woman", "Manyavar", "Jockey", "Spykar", "Pantaloons", "Max Fashion", "Killer Jeans", "Being Human"],
    "Electronics": ["Sony", "Samsung", "LG", "Panasonic", "Philips", "JBL", "Bose", "Sennheiser", "Micromax", "Lava", "boAt", "Noise", "Karbonn", "iBall", "Intex"],
    "Home": ["IKEA", "Crate & Barrel", "West Elm", "Williams Sonoma", "Pottery Barn", "Wayfair", "Godrej Interio", "Nilkamal", "Urban Ladder", "Pepperfry", "Fabindia", "Durian"],
    "Fitness": ["Nike", "Adidas", "Under Armour", "Reebok", "Peloton", "Bowflex", "NordicTrack", "Decathlon", "Cult.fit"],
    "Kids": ["Fisher-Price", "LEGO", "Carter's", "OshKosh B'gosh", "Gerber", "Graco", "Chicco", "Funskool", "Mothercare", "Toyzone"],
    "Beauty": ["L'Oréal", "Maybelline", "MAC Cosmetics", "Estée Lauder", "NARS", "Neutrogena", "Clinique", "Lakmé", "Nykaa", "Biotique", "VLCC", "Forest Essentials", "Lotus Herbals", "Colorbar", "Sugar Cosmetics", "Plum"],
    "Sports": ["Nike", "Adidas", "Under Armour", "Wilson", "Spalding", "Rawlings", "Puma", "Decathlon", "SG", "Cosco", "Nivia", "Vector X"],
    "Grocery": ["Kraft", "General Mills", "Nestlé", "Kellogg's", "Heinz", "Campbell's", "Britannia", "Parle", "Amul", "Haldiram's", "MDH", "ITC", "Tata Sampann", "Everest", "MTR"],
    "Toys": ["LEGO", "Hasbro", "Mattel", "Fisher-Price", "Playmobil", "Nerf", "Funskool", "Toyzone"],
    "Office": ["Staples", "HP", "Canon", "Epson", "Post-it", "Sharpie", "Pilot", "Camlin", "Cello", "Luxor", "Reynolds"],
    "Pets": ["Purina", "Pedigree", "Blue Buffalo", "KONG", "PetSafe", "Royal Canin", "Drools", "Heads Up For Tails"],
    "Automotive": ["Michelin", "Bosch", "Bridgestone", "Goodyear", "Castrol", "Mobil 1", "MRF", "Apollo Tyres", "Bajaj Auto", "TVS", "Hero MotoCorp", "Mahindra", "JK Tyre"],
    "Garden": ["Scotts", "Miracle-Gro", "Fiskars", "Black+Decker", "Husqvarna", "Ugaoo"],
    "Health": ["Johnson & Johnson", "Bayer", "Tylenol", "Advil", "Centrum", "Himalaya", "Dabur", "Patanjali", "Cipla", "Emami"],
    "Books": ["Penguin Random House", "HarperCollins", "Scholastic", "Simon & Schuster", "Macmillan", "Rupa Publications", "Amar Chitra Katha"],
    "Footwear": ["Nike", "Adidas", "New Balance", "Puma", "Vans", "Converse", "Skechers", "Reebok", "Bata", "Liberty", "Woodland", "Relaxo", "Metro Shoes", "Sparx", "Campus Shoes", "Paragon"],
    "Accessories": ["Fossil", "Ray-Ban", "Michael Kors", "Coach", "Kate Spade", "Titan", "Fastrack", "Da Milano", "Hidesign"],
    "Travel": ["Samsonite", "Away", "Tumi", "American Tourister", "Travelpro", "VIP", "Safari", "Skybags"],
    "Gaming": ["PlayStation", "Xbox", "Nintendo", "Razer", "Logitech G", "SteelSeries", "Ant Esports"],
    "Appliances": ["Whirlpool", "GE Appliances", "Samsung", "LG", "KitchenAid", "Bosch", "Frigidaire", "Godrej", "Voltas", "Havells", "Bajaj Electricals", "Usha", "Orient Electric", "IFB"],
    "Bedding": ["Tempur-Pedic", "Sealy", "Casper", "Purple", "Brooklinen", "Bombay Dyeing", "Spaces", "Welspun", "D'Decor"],
    "Kitchen": ["KitchenAid", "Cuisinart", "Instant Pot", "Le Creuset", "Ninja", "Vitamix", "Prestige", "Hawkins", "Pigeon", "Butterfly", "Bajaj Electricals", "Milton", "Wonderchef"],
    "Decor": ["West Elm", "CB2", "Pottery Barn", "Wayfair", "Anthropologie", "Pepperfry", "Urban Ladder", "Fabindia"],
    "Lighting": ["Philips Hue", "GE Lighting", "Lutron", "Feit Electric", "Havells", "Crompton", "Bajaj Electricals", "Syska", "Wipro Lighting"],
    "Storage": ["Rubbermaid", "Sterilite", "IKEA", "The Container Store", "Nilkamal", "Cello"],
    "Cleaning": ["Clorox", "Lysol", "Swiffer", "Dyson", "Bissell", "Shark", "Godrej", "Harpic", "Vim"],
    "Stationery": ["Sharpie", "Pilot", "Post-it", "Moleskine", "Crayola", "Camlin", "Classmate", "Navneet", "Cello", "Reynolds", "Flair"],
    "Baby": ["Pampers", "Huggies", "Gerber", "Fisher-Price", "Graco", "Chicco", "Himalaya", "LuvLap", "Mothercare"],
    "Luggage": ["Samsonite", "Tumi", "Away", "American Tourister", "VIP", "Safari", "Skybags"],
    "Audio": ["Bose", "Sony", "JBL", "Sennheiser", "Sonos", "Beats", "boAt", "Zebronics", "Noise", "Portronics"],
    "Wearables": ["Apple", "Fitbit", "Garmin", "Samsung", "Fossil", "boAt", "Noise", "Fire-Boltt", "Titan", "GOQii"],
    "Smart Home": ["Google Nest", "Amazon Ring", "Philips Hue", "Ecobee", "Wyze", "Syska"],
    "Cycling": ["Trek", "Specialized", "Cannondale", "Giant", "Schwinn", "Hero Cycles", "Firefox"],
    "Running": ["Nike", "Asics", "Brooks", "Saucony", "Hoka", "New Balance", "Decathlon"],
    "Camping": ["Coleman", "REI Co-op", "MSR", "Big Agnes", "YETI", "Wildcraft"],
    "Hiking": ["Merrell", "Salomon", "Osprey", "Columbia", "Black Diamond", "Wildcraft", "Decathlon"],
    "Fishing": ["Shimano", "Rapala", "Abu Garcia", "Berkley"],
    "Swimming": ["Speedo", "TYR", "Arena", "Nike"],
    "Yoga": ["Lululemon", "Manduka", "Gaiam", "Alo Yoga", "Decathlon"],
    "Skincare": ["CeraVe", "Neutrogena", "La Roche-Posay", "Olay", "The Ordinary", "Himalaya", "Biotique", "Mamaearth", "Patanjali", "Lotus Herbals", "Plum"],
    "Haircare": ["Pantene", "Head & Shoulders", "TRESemmé", "Dove", "Herbal Essences", "Dabur", "Patanjali", "Parachute", "Indulekha", "Bajaj Almond Drops"],
    "Supplements": ["Nature Made", "Optimum Nutrition", "GNC", "NOW Foods", "Patanjali", "MuscleBlaze", "HealthKart", "Zandu"],
    "Gadgets": ["Apple", "Samsung", "Anker", "Belkin", "Logitech", "boAt", "Micromax", "Portronics", "Ambrane"],
    "Computing": ["Dell", "HP", "Lenovo", "Apple", "Asus", "Acer", "iBall"],
    "Networking": ["Netgear", "TP-Link", "Linksys", "Asus", "Ubiquiti"],
    "Monitors": ["Dell", "LG", "Samsung", "Asus", "BenQ", "ViewSonic"],
    "Peripherals": ["Logitech", "Razer", "Corsair", "SteelSeries", "HyperX", "Zebronics", "iBall", "Ant Esports"],
    "Fragrance": ["Chanel", "Dior", "Calvin Klein", "Gucci", "Versace", "Fogg", "Wild Stone", "Park Avenue", "Engage", "Layer'r"],
    "Jewelry": ["Pandora", "Tiffany & Co.", "Swarovski", "Kay Jewelers", "Tanishq", "Kalyan Jewellers", "CaratLane", "Malabar Gold"],
}

# Sanity check at import time: every category referenced in CATEGORIES must
# have at least one brand, and every category in CATEGORY_BRANDS must be a
# real category. Fail loudly (at generation time) rather than silently.
_missing = [c for c in CATEGORIES if c not in CATEGORY_BRANDS or not CATEGORY_BRANDS[c]]
if _missing:
    raise RuntimeError(f"CATEGORY_BRANDS missing entries for: {_missing}")

# Build the deduplicated global brand list (preserves first-seen order) and a
# reverse map of brand_name -> sorted list of categories it belongs to. A
# brand naturally gets more categories if it legitimately appears in more of
# the CATEGORY_BRANDS lists above — no artificial cap.
_BRAND_NAME_TO_CATEGORIES: Dict[str, List[str]] = {}
for _cat in CATEGORIES:
    for _brand_name in CATEGORY_BRANDS[_cat]:
        _BRAND_NAME_TO_CATEGORIES.setdefault(_brand_name, []).append(_cat)

BRAND_NAMES: List[str] = list(_BRAND_NAME_TO_CATEGORIES.keys())  # unique, first-seen order

WAREHOUSES = [
    ("North", "US-North"),
    ("South", "US-South"),
    ("East", "US-East"),
    ("West", "US-West"),
    ("Central", "US-Central"),
]
WAREHOUSES = [(f"W{idx+1:03d}", f"Warehouse {name}", region) for idx, (name, region) in enumerate(WAREHOUSES * 5)][:25]

COLORS = ["Black", "White", "Blue", "Green", "Red", "Gray", "Navy", "Pink", "Brown", "Olive", "Teal", "Purple"]
SIZES = ["XS", "S", "M", "L", "XL", "XXL", "One Size"]
MATERIALS = ["Polyester", "Cotton", "Nylon", "Stainless Steel", "Aluminum", "Leather", "Glass", "Ceramic", "Silicone", "Wood", "Plastic", "Wool"]
TIER = ["Value", "Mid", "Premium"]
CHANNELS = ["web", "mobile", "marketplace", "store"]
ORDER_STATUSES = ["placed", "packed", "shipped", "out_for_delivery", "delivered", "cancelled", "returned"]
ITEM_STATUSES = ["fulfilled", "backordered", "cancelled", "returned"]
TRACKING_STATUSES = ["label_created", "in_transit", "out_for_delivery", "delivered", "exception"]
EVENT_TYPES = ["search", "view", "click", "add_to_cart", "wishlist", "purchase"]
TOOL_NAMES = ["inventory", "order_tracking", "recommendation", "cart", "memory"]
CARRIERS = ["BlueDart", "Delhivery", "DHL", "FedEx", "UPS", "USPS", "Aramex"]
FIRST_NAMES = ["Aarav", "Priya", "Isha", "Rohan", "Anika", "Kabir", "Meera", "Arjun", "Sara", "Nikhil", "Sana", "Vikram", "Neha", "Rahul", "Pooja", "Dev", "Kavya", "Anushka", "Aditya", "Simran"]
LAST_NAMES = ["Kapoor", "Sharma", "Verma", "Patel", "Iyer", "Gupta", "Reddy", "Nair", "Singh", "Malhotra", "Mehta", "Bose", "Chopra", "Agarwal", "Jain"]

# Real, category-appropriate product types (the actual "thing" being sold),
# not capped to a fixed size — a category can list as many product types as
# make sense. This is what fixes issues like "Electronics -> LG Backpack":
# the product's title word now always comes from the SAME category it's
# filed under, instead of a single generic cross-category list.
CATEGORY_PRODUCT_TYPES: Dict[str, List[str]] = {
    "Outdoor": ["Tent", "Camping Backpack", "Sleeping Bag", "Camping Chair", "Cooler", "Headlamp", "Hammock"],
    "Clothing": ["Jacket", "T-shirt", "Hoodie", "Jeans", "Sweater", "Shirt", "Dress"],
    "Electronics": ["Headphones", "Bluetooth Speaker", "Smartwatch", "Power Bank", "Wireless Earbuds", "Digital Camera"],
    "Home": ["Table Lamp", "Area Rug", "Curtains", "Wall Clock", "Throw Pillow", "Storage Bin"],
    "Fitness": ["Yoga Mat", "Resistance Bands", "Dumbbell Set", "Foam Roller", "Jump Rope"],
    "Kids": ["Building Blocks", "Stuffed Animal", "Puzzle", "Kids Backpack", "Ride-On Toy"],
    "Beauty": ["Lipstick", "Foundation", "Mascara", "Face Serum", "Compact Powder"],
    "Sports": ["Cricket Bat", "Football", "Basketball", "Tennis Racket", "Sports Bag"],
    "Grocery": ["Cookies", "Snack Pack", "Cereal", "Instant Noodles", "Tea Pack"],
    "Toys": ["Action Figure", "Board Game", "Remote Control Car", "Building Set"],
    "Office": ["Notebook", "Pen Set", "Desk Organizer", "Stapler", "File Folder"],
    "Pets": ["Dog Food", "Pet Bed", "Leash", "Chew Toy", "Cat Litter"],
    "Automotive": ["Car Cover", "Tire", "Engine Oil", "Dash Cam", "Car Charger"],
    "Garden": ["Garden Hose", "Pruning Shears", "Plant Pot", "Lawn Mower"],
    "Health": ["Multivitamin", "Digital Thermometer", "First Aid Kit", "Blood Pressure Monitor"],
    "Books": ["Novel", "Cookbook", "Biography", "Children's Book"],
    "Footwear": ["Boots", "Running Shoes", "Sandals", "Sneakers", "Loafers"],
    "Accessories": ["Wallet", "Sunglasses", "Belt", "Watch", "Scarf"],
    "Travel": ["Suitcase", "Travel Pillow", "Passport Holder", "Packing Cubes"],
    "Gaming": ["Game Controller", "Gaming Headset", "Gaming Mouse", "Gaming Chair"],
    "Appliances": ["Microwave", "Refrigerator", "Washing Machine", "Air Conditioner"],
    "Bedding": ["Blanket", "Pillow", "Bedsheet Set", "Comforter", "Mattress Protector"],
    "Kitchen": ["Pan", "Knife Set", "Coffee Maker", "Blender", "Mixer Grinder"],
    "Decor": ["Wall Art", "Vase", "Scented Candle", "Photo Frame"],
    "Lighting": ["Table Lamp", "LED Bulb", "Ceiling Light", "String Lights"],
    "Storage": ["Storage Box", "Closet Organizer", "Shelving Unit"],
    "Cleaning": ["Vacuum Cleaner", "Mop", "Cleaning Spray", "Broom"],
    "Stationery": ["Notebook", "Pen", "Sticky Notes", "Marker Set"],
    "Baby": ["Diapers", "Baby Wipes", "Baby Bottle", "Stroller"],
    "Luggage": ["Suitcase", "Duffel Bag", "Travel Backpack", "Garment Bag"],
    "Audio": ["Headphones", "Bluetooth Speaker", "Wireless Earbuds", "Soundbar"],
    "Wearables": ["Smartwatch", "Fitness Band", "Smart Ring"],
    "Smart Home": ["Smart Plug", "Smart Bulb", "Video Doorbell", "Smart Thermostat"],
    "Cycling": ["Bicycle", "Bike Helmet", "Bike Lock", "Cycling Gloves"],
    "Running": ["Running Shoes", "Running Shorts", "Running Watch"],
    "Camping": ["Tent", "Sleeping Bag", "Camping Stove", "Lantern"],
    "Hiking": ["Hiking Boots", "Trekking Pole", "Hiking Backpack"],
    "Fishing": ["Fishing Rod", "Fishing Reel", "Tackle Box", "Fishing Line"],
    "Swimming": ["Swimsuit", "Swim Goggles", "Swim Cap"],
    "Yoga": ["Yoga Mat", "Yoga Block", "Yoga Pants"],
    "Skincare": ["Face Serum", "Moisturizer", "Sunscreen", "Face Wash"],
    "Haircare": ["Shampoo", "Conditioner", "Hair Oil", "Hair Dryer"],
    "Supplements": ["Protein Powder", "Multivitamin", "Fish Oil"],
    "Gadgets": ["Power Bank", "Smart Plug", "Bluetooth Tracker"],
    "Computing": ["Laptop", "Desktop PC", "External Hard Drive"],
    "Networking": ["Wi-Fi Router", "Range Extender", "Ethernet Cable"],
    "Monitors": ["Monitor", "Monitor Stand"],
    "Peripherals": ["Mouse", "Keyboard", "Webcam"],
    "Fragrance": ["Perfume", "Eau de Toilette", "Body Mist"],
    "Jewelry": ["Necklace", "Earrings", "Bracelet", "Ring"],
}

_missing_types = [c for c in CATEGORIES if c not in CATEGORY_PRODUCT_TYPES or not CATEGORY_PRODUCT_TYPES[c]]
if _missing_types:
    raise RuntimeError(f"CATEGORY_PRODUCT_TYPES missing entries for: {_missing_types}")

# Categories whose products meaningfully come in sizes (apparel/footwear-like).
# Everything else gets "One Size".
SIZED_CATEGORIES = {"Clothing", "Footwear", "Running", "Swimming", "Yoga", "Kids"}

ROOT = Path(".")


def dt(days_back: int = 0, hours_back: int = 0) -> datetime:
    return datetime(2026, 7, 11, 12, 0, 0) - timedelta(days=days_back, hours=hours_back)


def stable_choice(seq: Sequence, idx: int):
    return seq[idx % len(seq)]


def chunked_range(n: int, chunk_size: int) -> Iterator[Tuple[int, int]]:
    for start in range(0, n, chunk_size):
        yield start, min(start + chunk_size, n)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, header: Sequence[str], rows: Iterable[Sequence], mode: str = "w") -> None:
    with path.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if mode == "w":
            writer.writerow(header)
        for row in rows:
            writer.writerow(row)


def p_id(i: int) -> str:
    return f"P{i:07d}"


def v_id(i: int) -> str:
    return f"V{i:08d}"


def c_id(i: int) -> str:
    return f"C{i:07d}"


def o_id(i: int) -> str:
    return f"O{i:08d}"


def oi_id(i: int) -> str:
    return f"OI{i:09d}"


def r_id(i: int) -> str:
    return f"R{i:08d}"


def e_id(i: int) -> str:
    return f"E{i:09d}"


def t_id(i: int) -> str:
    return f"T{i:08d}"


def s_id(i: int) -> str:
    return f"S{i:08d}"


def generate_categories(out: Path) -> List[Tuple[str, str, str, bool]]:
    rows = []
    header = ["category_id", "category_name", "parent_category_id", "is_age_sensitive"]
    for i, name in enumerate(CATEGORIES, start=1):
        cat_id = f"CAT{i:03d}"
        parent = f"CAT{((i - 1) // 7) + 1:03d}" if i > 7 else ""
        age_sensitive = name in {"Beauty", "Supplements", "Gadgets", "Fragrance", "Jewelry"}
        rows.append((cat_id, name, parent, age_sensitive))
    write_csv(out / "categories.csv", header, rows)
    return rows


def generate_brands(out: Path) -> List[Tuple[str, str, str, str]]:
    """Write one row per unique real brand name, tagged with every category
    it legitimately belongs to (pipe-separated, no fixed count)."""
    rows = []
    header = ["brand_id", "brand_name", "tier", "categories"]
    for i, name in enumerate(BRAND_NAMES, start=1):
        cats = "|".join(_BRAND_NAME_TO_CATEGORIES[name])
        rows.append((f"BR{i:03d}", name, TIER[i % len(TIER)], cats))
    write_csv(out / "brands.csv", header, rows)
    return rows


def generate_warehouses(out: Path) -> List[Tuple[str, str, str]]:
    header = ["warehouse_id", "warehouse_name", "region"]
    write_csv(out / "warehouses.csv", header, WAREHOUSES)
    return WAREHOUSES


# category_name -> list of brand_id strings (e.g. "BR014") that belong to it.
# Built once at import time from BRAND_NAMES order, so it lines up exactly
# with what generate_brands() writes to brands.csv.
CATEGORY_TO_BRAND_IDS: Dict[str, List[str]] = {cat: [] for cat in CATEGORIES}
for _idx, _name in enumerate(BRAND_NAMES, start=1):
    _bid = f"BR{_idx:03d}"
    for _cat in _BRAND_NAME_TO_CATEGORIES[_name]:
        CATEGORY_TO_BRAND_IDS[_cat].append(_bid)


def product_attributes(category: str, i: int) -> Tuple[str, str, bool, int, str, str]:
    title_base = stable_choice(CATEGORY_PRODUCT_TYPES[category], i)
    color = stable_choice(COLORS, i * 3 + 1)
    material = stable_choice(MATERIALS, i * 5 + 2)
    age_sensitive = category in {"Beauty", "Supplements", "Gadgets", "Fragrance", "Jewelry"}
    min_age = 18 if age_sensitive else 0
    description = (
        f"{color} {title_base.lower()} for {category.lower()} use with durable {material.lower()} construction. "
        f"Designed for search, compare, and recommendation workflows."
    )
    return title_base, description, age_sensitive, min_age, color, material


def generate_products(out: Path, n_products: int, rng: random.Random, chunk_size: int = 100_000):
    header = ["product_id", "category_id", "brand_id", "title", "description", "base_price", "rating_avg", "review_count", "age_restricted", "min_age", "color_family", "material", "created_at"]
    path = out / "products.csv"
    first = True
    for start, end in chunked_range(n_products, chunk_size):
        rows = []
        for i in range(start, end):
            category_id = f"CAT{(i % len(CATEGORIES)) + 1:03d}"
            category_name = CATEGORIES[i % len(CATEGORIES)]
            # Brand is now drawn only from brands that actually belong to
            # this category — deterministic cycling, no fixed-size cap.
            candidate_brand_ids = CATEGORY_TO_BRAND_IDS[category_name]
            brand_id = candidate_brand_ids[i % len(candidate_brand_ids)]
            brand_name = BRAND_NAMES[int(brand_id[2:]) - 1]
            title_base, description, age_sensitive, min_age, color, material = product_attributes(category_name, i)
            base_price = round(8 + ((i * 7) % 492) + ((i % 100) / 100), 2)
            rating_avg = round(2.8 + ((i * 17) % 22) / 10, 2)
            review_count = 5 + (i % 5000)
            created_at = (dt(days_back=(i % 1460)) - timedelta(hours=i % 24)).isoformat(sep=" ")
            rows.append((p_id(i), category_id, brand_id, f"{brand_name} {title_base}", description, base_price, rating_avg, review_count, age_sensitive, min_age, color, material, created_at))
        write_csv(path, header, rows, mode="w" if first else "a")
        first = False


def generate_variants(out: Path, n_products: int, n_variants: int, rng: random.Random, chunk_size: int = 100_000):
    header = ["variant_id", "product_id", "size_label", "color_name", "sku", "weight_grams", "is_active"]
    path = out / "product_variants.csv"
    first = True
    for start, end in chunked_range(n_variants, chunk_size):
        rows = []
        for i in range(start, end):
            product_idx = i % n_products
            # Recompute the same category the product actually belongs to
            # (must match generate_products' formula exactly) so sizing
            # logic is consistent with what the product really is.
            category_name = CATEGORIES[product_idx % len(CATEGORIES)]
            color = stable_choice(COLORS, i * 7 + 3)
            if category_name in SIZED_CATEGORIES:
                size = stable_choice(["XS", "S", "M", "L", "XL", "XXL"], i * 5 + 1)
            else:
                size = "One Size"
            sku = f"SKU-{product_idx:07d}-{i % 9999:04d}"
            weight = 100 + (i % 4000)
            rows.append((v_id(i), p_id(product_idx), size, color, sku, weight, (i % 17) != 0))
        write_csv(path, header, rows, mode="w" if first else "a")
        first = False


def generate_customers(out: Path, n_customers: int, rng: random.Random, chunk_size: int = 100_000):
    header = ["customer_id", "first_name", "last_name", "email", "signup_date", "home_region", "age"]
    path = out / "customers.csv"
    first = True
    regions = [r[2] for r in WAREHOUSES]
    for start, end in chunked_range(n_customers, chunk_size):
        rows = []
        for i in range(start, end):
            fn = stable_choice(FIRST_NAMES, i)
            ln = stable_choice(LAST_NAMES, i * 2 + 1)
            email = f"{fn.lower()}.{ln.lower()}{i % 1000:03d}@example.com"
            signup = (date(2022, 1, 1) + timedelta(days=i % 1280)).isoformat()
            region = stable_choice(regions, i)
            age = 18 + (i % 47)
            rows.append((c_id(i), fn, ln, email, signup, region, age))
        write_csv(path, header, rows, mode="w" if first else "a")
        first = False


def generate_preferences(out: Path, n_customers: int, chunk_size: int = 100_000):
    header = ["customer_id", "preferred_categories", "preferred_brands", "budget_max", "style_notes", "updated_at"]
    path = out / "customer_preferences.csv"
    first = True
    for start, end in chunked_range(n_customers, chunk_size):
        rows = []
        for i in range(start, end):
            cats = ",".join([CATEGORIES[(i + j) % len(CATEGORIES)] for j in range(3)])
            brands = ",".join([BRAND_NAMES[(i + j * 11) % len(BRAND_NAMES)] for j in range(3)])
            budget = round(25 + (i % 275), 2)
            note = ["minimal", "sporty", "technical", "casual", "premium", "budget", "outdoor", "travel"][(i * 3) % 8]
            updated = (dt(days_back=(i % 60), hours_back=(i % 24))).isoformat(sep=" ")
            rows.append((c_id(i), cats, brands, budget, note, updated))
        write_csv(path, header, rows, mode="w" if first else "a")
        first = False


def generate_order_items(out: Path, n_orders: int, n_items: int, n_products: int, n_variants: int, n_customers: int, chunk_size: int = 100_000) -> dict:
    header = ["order_item_id", "order_id", "customer_id", "product_id", "variant_id", "quantity", "unit_price", "discount_amount", "item_status"]
    path = out / "order_items.csv"
    first = True
    order_subtotals: dict[int, float] = {}
    for start, end in chunked_range(n_items, chunk_size):
        rows = []
        for i in range(start, end):
            order_idx = i % n_orders
            customer_idx = order_idx % n_customers
            variant_idx = i % n_variants
            product_idx = variant_idx % n_products
            quantity = 1 + (i % 3)
            unit_price = round(10 + ((product_idx * 7) % 490) + ((i % 100) / 100), 2)
            discount = round((i % 20) * 0.5, 2)
            item_status = stable_choice(ITEM_STATUSES, i)
            line_total = round(quantity * unit_price - discount, 2)
            order_subtotals[order_idx] = round(order_subtotals.get(order_idx, 0.0) + line_total, 2)
            rows.append((oi_id(i), o_id(order_idx), c_id(customer_idx), p_id(product_idx), v_id(variant_idx), quantity, unit_price, discount, item_status))
        write_csv(path, header, rows, mode="w" if first else "a")
        first = False
    return order_subtotals


def generate_orders(out: Path, n_orders: int, n_customers: int, order_subtotals: dict, rng: random.Random, chunk_size: int = 100_000):
    header = ["order_id", "customer_id", "order_date", "order_status", "channel", "subtotal", "shipping_fee", "tax_amount", "total_amount"]
    path = out / "orders.csv"
    first = True
    for start, end in chunked_range(n_orders, chunk_size):
        rows = []
        for i in range(start, end):
            cust = c_id(i % n_customers)
            order_date = (dt(days_back=(i % 365), hours_back=(i % 24))).isoformat(sep=" ")
            status = stable_choice(ORDER_STATUSES, i)
            channel = stable_choice(CHANNELS, i * 2 + 1)
            subtotal = order_subtotals.get(i, 0.0)
            shipping = round(3 + (i % 15), 2)
            tax = round(subtotal * 0.08, 2)
            total = round(subtotal + shipping + tax, 2)
            rows.append((o_id(i), cust, order_date, status, channel, subtotal, shipping, tax, total))
        write_csv(path, header, rows, mode="w" if first else "a")
        first = False


def generate_inventory(out: Path, n_variants: int, n_products: int, n_snapshots: int, chunk_size: int = 100_000):
    header = ["snapshot_id", "warehouse_id", "product_id", "variant_id", "snapshot_date", "stock_qty", "reserved_qty", "available_qty", "backorder_qty", "restock_eta_date"]
    path = out / "inventory_snapshots.csv"
    first = True
    for start, end in chunked_range(n_snapshots, chunk_size):
        rows = []
        for i in range(start, end):
            variant_idx = i % n_variants
            product_idx = variant_idx % n_products
            warehouse_id = WAREHOUSES[i % len(WAREHOUSES)][0]
            snapshot_date = (date(2026, 1, 1) + timedelta(days=i % 181)).isoformat()
            stock = (i * 11) % 250
            reserved = i % 30
            available = max(stock - reserved, 0)
            backorder = 0 if available > 0 else (i % 5)
            restock_eta = (date(2026, 1, 1) + timedelta(days=(i % 14) + 1)).isoformat() if available == 0 else ""
            rows.append((s_id(i), warehouse_id, p_id(product_idx), v_id(variant_idx), snapshot_date, stock, reserved, available, backorder, restock_eta))
        write_csv(path, header, rows, mode="w" if first else "a")
        first = False


def generate_tracking(out: Path, n_orders: int, chunk_size: int = 100_000):
    header = ["order_id", "carrier", "tracking_status", "shipped_date", "expected_delivery_date", "delivered_date", "last_event_at"]
    path = out / "order_tracking.csv"
    first = True
    for start, end in chunked_range(n_orders, chunk_size):
        rows = []
        for i in range(start, end):
            status = stable_choice(TRACKING_STATUSES, i)
            shipped = (date(2026, 1, 1) + timedelta(days=i % 120)).isoformat() if status != "label_created" else ""
            expected = (date(2026, 1, 1) + timedelta(days=(i % 120) + 5)).isoformat()
            delivered = (date(2026, 1, 1) + timedelta(days=(i % 120) + 7)).isoformat() if status == "delivered" else ""
            last_event = (dt(days_back=(i % 120), hours_back=(i % 24))).isoformat(sep=" ")
            rows.append((o_id(i), stable_choice(CARRIERS, i), status, shipped, expected, delivered, last_event))
        write_csv(path, header, rows, mode="w" if first else "a")
        first = False


def generate_reviews(out: Path, n_reviews: int, n_order_items: int, n_orders: int, n_customers: int, n_products: int, n_variants: int, chunk_size: int = 100_000) -> dict:
    header = ["review_id", "order_item_id", "order_id", "customer_id", "product_id", "rating", "review_title", "review_body", "review_date", "verified_purchase"]
    path = out / "reviews.csv"
    first = True
    product_stats: dict[int, list] = {}
    for start, end in chunked_range(n_reviews, chunk_size):
        rows = []
        for i in range(start, end):
            order_item_idx = i % n_order_items
            order_idx = order_item_idx % n_orders
            customer_idx = order_idx % n_customers
            variant_idx = order_item_idx % n_variants
            product_idx = variant_idx % n_products
            rating = 1 + (i % 5)
            title = ["Great fit", "Good value", "Solid quality", "Nice style", "Works well"][i % 5]
            body = f"Synthetic review {i} for product {product_idx:07d}. This item matched the stated use case and budget constraints."
            review_date = (date(2026, 1, 1) + timedelta(days=i % 365)).isoformat()
            rows.append((r_id(i), oi_id(order_item_idx), o_id(order_idx), c_id(customer_idx), p_id(product_idx), rating, title, body, review_date, True))

            if product_idx not in product_stats:
                product_stats[product_idx] = [0, 0]
            product_stats[product_idx][0] += rating
            product_stats[product_idx][1] += 1
        write_csv(path, header, rows, mode="w" if first else "a")
        first = False
    return product_stats

def generate_events(out: Path, n_events: int, n_customers: int, n_products: int, n_variants: int, chunk_size: int = 100_000):
    header = ["event_id", "customer_id", "session_id", "event_type", "query_text", "product_id", "variant_id", "event_ts", "dwell_ms"]
    path = out / "behavior_events.csv"
    first = True
    query_templates = [
        "waterproof jacket under 80",
        "cheap hiking boots for cold weather",
        "similar but cheaper",
        "does it come in green",
        "size M in stock",
        "order tracking last Tuesday",
        "budget hiking gear",
        "lightweight backpack",
    ]
    for start, end in chunked_range(n_events, chunk_size):
        rows = []
        for i in range(start, end):
            cust = c_id(i % n_customers)
            session = f"S{i // 8:010d}"
            etype = stable_choice(EVENT_TYPES, i)
            query = query_templates[i % len(query_templates)] if etype in {"search", "view"} else ""

            if etype in {"click", "add_to_cart", "purchase"}:
                variant_idx = i % n_variants
                product_idx = variant_idx % n_products
            else:
                variant_idx = ""
                product_idx = i % n_products

            event_ts = (dt(days_back=(i % 90), hours_back=(i % 24))).isoformat(sep=" ")
            dwell = 500 + (i % 120000)
            rows.append((e_id(i), cust, session, etype, query, p_id(product_idx), v_id(int(variant_idx)) if variant_idx != "" else "", event_ts, dwell))
        write_csv(path, header, rows, mode="w" if first else "a")
        first = False


def generate_tool_logs(out: Path, n_logs: int, n_customers: int, chunk_size: int = 100_000):
    header = ["call_id", "customer_id", "session_id", "tool_name", "success", "latency_ms", "error_type", "called_at", "payload_summary"]
    path = out / "tool_call_logs.csv"
    first = True
    errors = ["timeout", "rate_limited", "upstream_5xx", "invalid_request", "none"]
    for start, end in chunked_range(n_logs, chunk_size):
        rows = []
        for i in range(start, end):
            success = (i % 13) != 0
            err = "" if success else stable_choice(errors[:-1], i)
            rows.append((t_id(i), c_id(i % n_customers), f"S{i // 5:010d}", stable_choice(TOOL_NAMES, i), success, 80 + (i % 1200), err, (dt(days_back=(i % 14), hours_back=(i % 24))).isoformat(sep=" "), f"{{'table':'{stable_choice(TOOL_NAMES, i)}'}}"))
        write_csv(path, header, rows, mode="w" if first else "a")
        first = False


def scale_counts(counts: dict, scale: float) -> dict:
    if scale <= 0:
        raise ValueError("scale must be positive")
    if scale == 1:
        return counts
    scaled = {}
    for k, v in counts.items():
        if k in {"categories", "brands", "warehouses"}:
            scaled[k] = v
        else:
            scaled[k] = max(1, int(math.ceil(v * scale)))
    return scaled


def enforce_full_product_coverage(counts: dict) -> dict:
    """Guarantee every product gets at least one variant, and every product
    gets at least one order_item (i.e. shows up in orders).

    Why this is needed: variants are assigned to products via
    `product_idx = i % n_products` in generate_variants, and order_items are
    assigned via `variant_idx = i % n_variants` then `product_idx = variant_idx
    % n_products` in generate_order_items. Those formulas only cover every
    product_idx in [0, n_products) if:
        product_variants >= products
        order_items       >= product_variants
    If either count is smaller, the tail of products simply never gets
    picked and ends up with zero variants / zero orders. Rather than let
    that happen silently, we bump the counts up here and tell you when we do.
    """
    adjusted = dict(counts)
    notes = []

    if adjusted["product_variants"] < adjusted["products"]:
        notes.append(
            f"product_variants ({adjusted['product_variants']}) < products ({adjusted['products']}); "
            f"raising product_variants to {adjusted['products']} so every product gets at least one variant."
        )
        adjusted["product_variants"] = adjusted["products"]

    if adjusted["order_items"] < adjusted["product_variants"]:
        notes.append(
            f"order_items ({adjusted['order_items']}) < product_variants ({adjusted['product_variants']}); "
            f"raising order_items to {adjusted['product_variants']} so every product appears in at least one order."
        )
        adjusted["order_items"] = adjusted["product_variants"]

    if notes:
        print("NOTE: adjusted counts to guarantee full product coverage in orders:")
        for n in notes:
            print("  -", n)

    return adjusted


def patch_product_review_stats(out: Path, product_stats: dict) -> None:
    src = out / "products.csv"
    tmp = out / "products.csv.tmp"
    with src.open("r", newline="", encoding="utf-8") as fin, tmp.open("w", newline="", encoding="utf-8") as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout)
        header = next(reader)
        writer.writerow(header)
        rating_idx = header.index("rating_avg")
        count_idx = header.index("review_count")
        product_id_idx = header.index("product_id")
        for row in reader:
            product_idx = int(row[product_id_idx][1:])
            if product_idx in product_stats:
                rating_sum, count = product_stats[product_idx]
                row[rating_idx] = round(rating_sum / count, 2)
                row[count_idx] = count
            else:
                row[rating_idx] = 0.0
                row[count_idx] = 0
            writer.writerow(row)
    tmp.replace(src)

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the raw synthetic retail warehouse CSVs. "
                     "For LLM-based description/review enrichment, run "
                     "enrich_descriptions.py / enrich_reviews.py afterward."
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--format", choices=["csv"], default="csv")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--counts", default=None, help="Optional JSON counts file")
    args = parser.parse_args()

    out = Path(args.out)
    ensure_dir(out)
    rng = random.Random(args.seed)

    if args.counts:
        with open(args.counts, "r", encoding="utf-8") as f:
            counts = json.load(f)
    else:
        with open(Path(__file__).with_name("counts.json"), "r", encoding="utf-8") as f:
            counts = json.load(f)

    counts = scale_counts(counts, args.scale)
    counts = enforce_full_product_coverage(counts)

    generate_categories(out)
    generate_brands(out)
    generate_warehouses(out)
    generate_products(out, counts["products"], rng)
    generate_variants(out, counts["products"], counts["product_variants"], rng)
    generate_customers(out, counts["customers"], rng)
    #generate_preferences(out, counts["customer_preferences"])
    order_subtotals = generate_order_items(out, counts["orders"], counts["order_items"], counts["products"], counts["product_variants"], counts["customers"])
    generate_orders(out, counts["orders"], counts["customers"], order_subtotals, rng)
    generate_inventory(out, counts["product_variants"], counts["products"], counts["inventory_snapshots"])
    generate_tracking(out, counts["orders"])
    product_stats = generate_reviews(out, counts["reviews"], counts["order_items"], counts["orders"], counts["customers"], counts["products"], counts["product_variants"])
    patch_product_review_stats(out, product_stats)
    #generate_events(out, counts["behavior_events"], counts["customers"], counts["products"], counts["product_variants"])
    #generate_tool_logs(out, counts["tool_call_logs"], counts["customers"])

    print(json.dumps({"output": str(out.resolve()), "counts": counts}, indent=2))


if __name__ == "__main__":
    main()