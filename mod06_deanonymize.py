import pandas as pd

def load_data(anonymized_path, auxiliary_path):
    """
    Load anonymized and auxiliary datasets.
    """
    anon = pd.read_csv(anonymized_path)
    aux = pd.read_csv(auxiliary_path)
    return anon, aux


def link_records(anon_df, aux_df):
    """
    Attempt to link anonymized records to auxiliary records
    using exact matching on quasi-identifiers.

    Returns a DataFrame with columns:
      anon_id, matched_name
    containing ONLY uniquely matched records.
    """
    
    quasi_ids = ['age', 'zip3', 'gender']
    merged = anon_df.merge(aux_df, on=quasi_ids, how='inner')
    # Keep only (age, zip3, gender) that appear exactly once in each dataset
    anon_counts = anon_df.groupby(quasi_ids).size()
    aux_counts = aux_df.groupby(quasi_ids).size()
    unique_in_both = anon_counts[anon_counts == 1].index.intersection(
        aux_counts[aux_counts == 1].index
    )
    mask = merged.set_index(quasi_ids).index.isin(unique_in_both)
    result = merged.loc[mask, ['anon_id', 'name']].rename(columns={'name': 'matched_name'})
    return result


def deanonymization_rate(matches_df, anon_df):
    """
    Compute the fraction of anonymized records
    that were uniquely re-identified.
    """
    return len(matches_df) / len(anon_df)
