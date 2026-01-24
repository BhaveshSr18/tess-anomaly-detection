import lightkurve as lk
import numpy as np
from tqdm import tqdm
import os
import warnings

# Suppress non-critical warnings
warnings.filterwarnings('ignore', category=UserWarning)

SAVE_DIR = r"D:\Astrophysics Project\Astro-Project\data\raw"
os.makedirs(SAVE_DIR, exist_ok=True)

def download_random_stars(n=500):
    # Use known popular TESS targets with higher success rate
    # TIC IDs of known bright stars with TESS data
    known_tics = [25155310, 307210830, 402026209, 261136679, 231663901, 
                  176986225, 207470147, 207539675, 38846515, 114616042]
    
    # Mix with random IDs but start from higher range where TIC catalog is denser
    random_tics = np.random.randint(200000000, 600000001, size=n-len(known_tics))
    tic_ids = np.concatenate([known_tics, random_tics])

    success_count = 0
    failed_tics = []
    
    for tic in tqdm(tic_ids, desc="Downloading light curves"):
        try:
            # Set timeout and limit search with mission filter to speed up queries
            search = lk.search_lightcurve(f"TIC {tic}", mission="TESS")
            if len(search) == 0:
                continue

            # Download only the first result to avoid multiple file warning
            lc = search[0].download(quality_bitmask="hardest")
            
            # Verify the light curve has data
            if lc is None or len(lc) == 0:
                failed_tics.append((tic, "Empty light curve"))
                continue
            
            filename = f"{SAVE_DIR}/TIC_{tic}.fits"
            lc.to_fits(filename, overwrite=True)
            success_count += 1

        except Exception as e:
            failed_tics.append((tic, str(e)))
            continue
    
    print(f"\nSuccessfully downloaded {success_count} light curves out of {n} attempts")
    if failed_tics:
        print(f"Failed TICs (showing first 5): {failed_tics[:5]}")

if __name__ == "__main__":
    download_random_stars(500)
