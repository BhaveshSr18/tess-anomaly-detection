import numpy as np
import os
from tqdm import tqdm
from astropy.io import fits

RAW_DIR = r"D:\Astrophysics Project\Astro-Project\data\raw"
OUT_DIR = r"D:\Astrophysics Project\Astro-Project\data\processed"

os.makedirs(OUT_DIR, exist_ok=True)

FIXED_LENGTH = 2000

def process_file(path):
    """Load and process a TESS light curve FITS file."""
    # Read FITS file directly to avoid unit conversion issues
    with fits.open(path) as hdul:
        # Get the light curve extension (usually index 1 for TESS data)
        if len(hdul) > 1:
            data = hdul[1].data
        else:
            data = hdul[0].data
        
        # Extract flux - try different column names
        flux = None
        for col_name in ['FLUX', 'SAP_FLUX', 'PDCSAP_FLUX', 'sap_flux', 'pdcsap_flux', 'flux']:
            if col_name in data.dtype.names:
                flux = data[col_name]
                break
        
        if flux is None:
            raise ValueError(f"Could not find flux column in FITS file. Available columns: {data.dtype.names}")
        
        # Convert to float and handle any non-numeric values
        flux = np.asarray(flux, dtype=np.float32)
        
        # Remove NaN values
        flux = flux[~np.isnan(flux)]
        
        if len(flux) == 0:
            raise ValueError("No valid flux data after removing NaNs")
        
        # Remove outliers (simple sigma clipping)
        mean = np.nanmean(flux)
        std = np.nanstd(flux)
        if std > 0:
            mask = np.abs(flux - mean) < 5 * std
            flux = flux[mask]
        
        # Normalize if possible
        mean_flux = np.nanmean(np.abs(flux))
        if mean_flux > 1e-10:
            flux = flux / mean_flux
        
        # Pad or truncate to fixed length
        if len(flux) < FIXED_LENGTH:
            flux = np.pad(flux, (0, FIXED_LENGTH - len(flux)), mode='constant', constant_values=0)
        else:
            flux = flux[:FIXED_LENGTH]
        
        return flux.astype(np.float32)

def main():
    data = []
    failed_files = []
    success_count = 0

    files = sorted([f for f in os.listdir(RAW_DIR) if f.endswith(".fits")])
    
    for fname in tqdm(files, desc="Processing FITS files"):
        try:
            arr = process_file(os.path.join(RAW_DIR, fname))
            data.append(arr)
            success_count += 1
        except Exception as e:
            failed_files.append((fname, str(e)))
            continue

    if success_count == 0:
        print("ERROR: No files were successfully processed!")
        print(f"Failed files: {failed_files[:5]}")  # Show first 5 errors
        return
    
    data = np.array(data)
    print(f"\nSuccessfully processed {success_count}/{len(files)} files")
    print(f"Output shape: {data.shape}")
    
    if len(failed_files) > 0:
        print(f"Failed to process {len(failed_files)} files:")
        for fname, error in failed_files[:3]:
            print(f"  {fname}: {error}")
    
    np.save(f"{OUT_DIR}/lightcurves.npy", data)
    print(f"Saved to {OUT_DIR}/lightcurves.npy")

if __name__ == "__main__":
    main()
