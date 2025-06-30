import typing as t
import numpy as np
import pandas as pd


def generate_marketing_data(
    n_samples: int = 1000,
    n_campaigns: int = 5,
    n_user_features: int = 10,
    random_state: t.Optional[int] = None,
) -> pd.DataFrame:
    if random_state is not None:
        np.random.seed(random_state)

    user_features = np.random.randn(n_samples, n_user_features)
    campaigns = np.random.randint(0, n_campaigns, size=n_samples)
    campaign_params = np.random.randn(n_campaigns, n_user_features)
    conversion_probs = np.zeros(n_samples)
    for i in range(n_samples):
        logit = np.dot(user_features[i], campaign_params[campaigns[i]])
        conversion_probs[i] = 1.0 / (1.0 + np.exp(-logit))

    conversions = np.random.binomial(1, conversion_probs)
    unsubscribe_probs = 0.01 + 0.1 * (1 - conversion_probs)
    unsubscribes = np.random.binomial(1, unsubscribe_probs)

    df = pd.DataFrame()
    for i in range(n_user_features):
        df[f"user_feature_{i}"] = user_features[:, i]
    df["age_group"] = np.random.choice(
        ["18-24", "25-34", "35-44", "45-54", "55+"], size=n_samples
    )
    df["gender"] = np.random.choice(["M", "F", "Other"], size=n_samples)
    df["device"] = np.random.choice(["Mobile", "Desktop", "Tablet"], size=n_samples)
    df["previous_purchases"] = np.random.poisson(2, size=n_samples)
    df["campaign_id"] = [f"campaign_{c}" for c in campaigns]
    df["email_opened"] = np.random.binomial(1, 0.3 + 0.4 * conversion_probs)
    df["email_clicked"] = (
        np.random.binomial(1, 0.2 + 0.6 * conversion_probs)
        * df["email_opened"]  # only clicked if opened
    )
    df["conversion"] = conversions
    df["unsubscribe"] = unsubscribes
    df["day_of_week"] = np.random.randint(0, 7, size=n_samples)
    df["hour_of_day"] = np.random.randint(0, 24, size=n_samples)

    return df
