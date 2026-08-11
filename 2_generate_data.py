"""
generate_data.py
-----------------
Creates a synthetic but realistic retail customer dataset with 28 columns
covering demographics, purchase behavior, channel usage, engagement, and
category spend. Five latent (hidden) customer archetypes are baked in with
noise so the dataset behaves like real-world retail data: clusters exist,
but are not perfectly separable -- exactly what K-Means + PCA are meant to
uncover WITHOUT being told the labels.

The "true_segment" column is saved separately (not given to the clustering
pipeline) purely so we can sanity-check our unsupervised results afterward.
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 2000

# ---------------------------------------------------------------
# Define 5 latent archetypes with distinct behavioral centers.
# Each tuple below: (share_of_population, param dict of means)
# ---------------------------------------------------------------
archetypes = {
    "Premium Loyalist": dict(
        share=0.18, age=45, income=110, tenure=48, recency=8, freq=28,
        aov=145, web_v=10, deals=3, discount_rate=0.10, abandon=0.08,
        return_rate=0.03, complaints=0.2, loyalty=4200, email_open=0.55,
        app_min=90, social=35, wishlist=6, reviews=5, cs_contacts=0.5,
    ),
    "Bargain Hunter": dict(
        share=0.22, age=38, income=58, tenure=30, recency=15, freq=22,
        aov=38, web_v=14, deals=14, discount_rate=0.65, abandon=0.35,
        return_rate=0.12, complaints=0.8, loyalty=1400, email_open=0.40,
        app_min=60, social=40, wishlist=12, reviews=2, cs_contacts=1.2,
    ),
    "Digital Native": dict(
        share=0.20, age=26, income=52, tenure=14, recency=10, freq=16,
        aov=55, web_v=25, deals=6, discount_rate=0.30, abandon=0.45,
        return_rate=0.15, complaints=0.6, loyalty=900, email_open=0.25,
        app_min=180, social=85, wishlist=18, reviews=3, cs_contacts=0.7,
    ),
    "At-Risk / Dormant": dict(
        share=0.20, age=50, income=64, tenure=60, recency=140, freq=3,
        aov=48, web_v=2, deals=1, discount_rate=0.20, abandon=0.20,
        return_rate=0.20, complaints=2.1, loyalty=300, email_open=0.05,
        app_min=8, social=5, wishlist=1, reviews=0, cs_contacts=2.5,
    ),
    "Steady Average": dict(
        share=0.20, age=41, income=71, tenure=32, recency=35, freq=11,
        aov=62, web_v=8, deals=4, discount_rate=0.25, abandon=0.20,
        return_rate=0.08, complaints=0.5, loyalty=1600, email_open=0.30,
        app_min=35, social=20, wishlist=5, reviews=1, cs_contacts=0.8,
    ),
}

rows = []
segment_labels = []
names = list(archetypes.keys())
shares = [archetypes[n]["share"] for n in names]
assigned = np.random.choice(names, size=N, p=shares)

def noisy(mean, pct_sd=0.28, floor=0):
    val = np.random.normal(mean, abs(mean) * pct_sd + 1e-6)
    return max(floor, val)

for seg in assigned:
    p = archetypes[seg]
    age = noisy(p["age"], 0.18, 18)
    income = noisy(p["income"], 0.25, 15)
    tenure = noisy(p["tenure"], 0.30, 1)
    recency = noisy(p["recency"], 0.40, 0)
    freq = noisy(p["freq"], 0.35, 0)
    aov = noisy(p["aov"], 0.30, 5)
    monetary = freq * aov * np.random.normal(1.0, 0.1)

    web_v = noisy(p["web_v"], 0.35, 0)
    deals = min(freq, noisy(p["deals"], 0.4, 0))
    web_purch = freq * np.random.uniform(0.3, 0.6)
    store_purch = freq * np.random.uniform(0.2, 0.5)
    cat_purch = max(0, freq - web_purch - store_purch)

    discount_rate = np.clip(noisy(p["discount_rate"], 0.3), 0, 1)
    abandon = np.clip(noisy(p["abandon"], 0.3), 0, 1)
    return_rate = np.clip(noisy(p["return_rate"], 0.3), 0, 1)
    complaints = max(0, np.random.poisson(max(p["complaints"], 0.05)))
    loyalty = noisy(p["loyalty"], 0.3, 0)
    email_open = np.clip(noisy(p["email_open"], 0.3), 0, 1)
    app_min = noisy(p["app_min"], 0.4, 0)
    social = noisy(p["social"], 0.4, 0)
    wishlist = max(0, np.random.poisson(max(p["wishlist"], 0.1)))
    reviews = max(0, np.random.poisson(max(p["reviews"], 0.1)))
    cs_contacts = max(0, np.random.poisson(max(p["cs_contacts"], 0.05)))

    # category spend splits (share of monetary total across 5 categories,
    # weighted differently per archetype to add extra realistic structure)
    if seg == "Premium Loyalist":
        w = np.random.dirichlet([3, 3, 1, 3, 2])
    elif seg == "Bargain Hunter":
        w = np.random.dirichlet([1, 2, 4, 1, 1])
    elif seg == "Digital Native":
        w = np.random.dirichlet([3, 3, 1, 1, 3])
    elif seg == "At-Risk / Dormant":
        w = np.random.dirichlet([1, 1, 3, 1, 1])
    else:
        w = np.random.dirichlet([2, 2, 2, 2, 2])
    spend_elec, spend_appar, spend_groc, spend_home, spend_beauty = w * monetary

    rows.append(dict(
        Age=round(age, 1),
        AnnualIncome_k=round(income, 1),
        Tenure_Months=round(tenure, 1),
        Recency_Days=round(recency, 1),
        Frequency_Purchases=round(freq, 1),
        MonetaryTotal=round(monetary, 2),
        AvgOrderValue=round(aov, 2),
        NumWebPurchases=round(web_purch, 1),
        NumStorePurchases=round(store_purch, 1),
        NumCatalogPurchases=round(cat_purch, 1),
        NumWebVisitsPerMonth=round(web_v, 1),
        NumDealsPurchases=round(deals, 1),
        DiscountUsageRate=round(discount_rate, 3),
        CartAbandonRate=round(abandon, 3),
        ReturnRate=round(return_rate, 3),
        NumComplaints=complaints,
        LoyaltyPoints=round(loyalty, 0),
        EmailOpenRate=round(email_open, 3),
        AppSessionMinPerMonth=round(app_min, 1),
        SocialEngagementScore=round(social, 1),
        WishlistItems=wishlist,
        ReviewsWritten=reviews,
        CustomerServiceContacts=cs_contacts,
        Spend_Electronics=round(spend_elec, 2),
        Spend_Apparel=round(spend_appar, 2),
        Spend_Grocery=round(spend_groc, 2),
        Spend_HomeGoods=round(spend_home, 2),
        Spend_Beauty=round(spend_beauty, 2),
    ))
    segment_labels.append(seg)

df = pd.DataFrame(rows)
df.insert(0, "CustomerID", [f"CUST{i:05d}" for i in range(1, N + 1)])

df.to_csv("retail_customers.csv", index=False)
pd.Series(segment_labels, name="true_segment_HIDDEN").to_csv(
    "true_segments_HIDDEN_for_validation_only.csv", index=False
)

print(f"Generated {len(df)} customers x {df.shape[1]} columns")
print(df.head())
print("\nColumn list:", list(df.columns))
