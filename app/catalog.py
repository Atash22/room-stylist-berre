"""
catalog.py
==========
Product lookup keyed by predicted style. Entries below are real Berre.ca
products with verified image URLs, prices, and product page links (pulled
directly from berre.ca product pages).

NOTE (demo scope): only 3 products are fully verified so far, so some
style buckets still share a fallback pair rather than having a unique
match. To expand further: visit the product page on berre.ca, grab the
"src" of the main product image, and add an entry following the same
shape as the ones below.
"""

MID_CENTURY_PRODUCTS = [
    {
        "name": "Diana Boucle Sofa",
        "price": "$1,799.00",
        "url": "https://berre.ca/products/diana-boucle-sofa",
        "image": "https://berre.ca/cdn/shop/products/diana-boucle-sofa-439683.webp?v=1752263119&width=800",
    },
]

TRADITIONAL_PRODUCTS = [
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

# Demo-scope fallback: buckets without a unique verified match yet reuse
# the closest available real products rather than showing nothing.
CATALOG = {
    "mid_century_modern": MID_CENTURY_PRODUCTS,
    "contemporary_scandinavian": MID_CENTURY_PRODUCTS,  # closest visual match available
    "traditional_classic": TRADITIONAL_PRODUCTS,
    "rustic_farmhouse": TRADITIONAL_PRODUCTS,
    "coastal_tropical": TRADITIONAL_PRODUCTS,
    "eclectic_industrial": TRADITIONAL_PRODUCTS,
}


def get_recommendations(style: str, top_k: int = 3):
    return CATALOG.get(style, [])[:top_k]
