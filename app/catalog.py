"""
catalog.py
==========
Minimal product lookup keyed by predicted style. In a real build this
would be a scraped/exported snapshot of the Berre.ca catalog (Shopify
Storefront API), but for the course project a small hand-labeled JSON
is enough to demo the full pipeline end-to-end.
"""

"""
catalog.py
==========
Product lookup keyed by predicted style. Entries below are real Berre.ca
products with verified image URLs, prices, and product page links (pulled
directly from berre.ca product pages).

NOTE: for demo scope, two fully-verified products are used across all
style buckets. To expand: visit the product page on berre.ca, grab the
"src" of the main product image, and add an entry following the same
shape.
"""

REAL_PRODUCTS = [
    {
        "name": "Montego Living Room Set",
        "price": "$5,999.00",
        "url": "https://berre.ca/products/bellona-188",
        "image": "https://berre.ca/cdn/shop/files/montego-living-room-set-sofa-loveseat-armchair-by-bellona-359979.jpg?v=1738632201&width=800",
    },
    {
        "name": "Mirante Living Room Set (Beatto Light Brown)",
        "price": "$4,850.00",
        "url": "https://berre.ca/products/mirante-living-room-set-sofa",
        "image": "https://berre.ca/cdn/shop/files/mirante-living-room-set-sofa-and-loveseat-beatto-light-brown-by-bellona-508747.jpg?v=1738604901&width=800",
    },
]

CATALOG = {
    "mid_century_modern": REAL_PRODUCTS,
    "contemporary_scandinavian": REAL_PRODUCTS,
    "traditional_classic": REAL_PRODUCTS,
    "rustic_farmhouse": REAL_PRODUCTS,
    "coastal_tropical": REAL_PRODUCTS,
    "eclectic_industrial": REAL_PRODUCTS,
}


def get_recommendations(style: str, top_k: int = 3):
    return CATALOG.get(style, [])[:top_k]

