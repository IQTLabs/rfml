import re
import numpy as np

SAMPLE_FILENAME_RE = re.compile(r"^.*?(\d+)_(\d+)Hz.*\D(\d+)sps\.(c*[fisu]\d+|raw).*$")
#ALT_SAMPLE_FILENAME_RE = re.compile(r"^.*?([\d.]+)e(\d)_(\d+)_.*_(\d+)\.(c*[fisu]\d+|raw).*$")
ALT_SAMPLE_FILENAME_RE = re.compile(r"^.*?_*([\d.e]+)_(\d+)_([\d-]+)_(\d+)s_(\d+)\.(c*[fisu]\d+|raw).*$") # frequency_center, sampling_rate, gain, duration, timestamp, datatype

SAMPLE_DTYPES = {
    "s8": ("<i1", "signed-integer"),
    "s16": ("<i2", "signed-integer"),
    "s32": ("<i4", "signed-integer"),
    "u8": ("<u1", "unsigned-integer"),
    "u16": ("<u2", "unsigned-integer"),
    "u32": ("<u4", "unsigned-integer"),
    "raw": ("<f4", "float"),
}
SIGMF_DTYPES = {
    "s8": "ci8",
    "s16": "ci16_le",
    "s32": "ci32_le",
    "u8": "cu8",
    "u16": "cu16_le",
    "u32": "cu32_le",
    "raw": "cf32_le",
    "ci16": "ci16_le",
}


def parse_zst_filename(filename):
    """
    Parses metadata from .zst filenames following the specification used by GamutRF.

    Args:
        filename (str): Filename to parse.

    Returns:
        dict: Contains metadata regarding the I/Q recording.
    """
    parsed = False
    
    match = SAMPLE_FILENAME_RE.match(filename)
    try:
        timestamp = int(match.group(1))
        freq_center = int(match.group(2))
        sample_rate = int(match.group(3))
        sample_type = match.group(4)
        parsed = True
    except AttributeError:
        pass

    match = ALT_SAMPLE_FILENAME_RE.match(filename)
    # frequency_center, sampling_rate, gain, duration, timestamp, datatype
    try: 
        freq_center = int(float(match.group(1)))
        sample_rate = int(match.group(2))
        gain = int(match.group(3))
        duration = int(match.group(4))
        timestamp = int(match.group(5))
        sample_type = match.group(6)
        parsed = True
    except AttributeError:
        pass
        
    if not parsed:
        
        #raise ValueError(f"Could not parse file {filename}")
        print(f"Could not parse file {filename}")
        return None
        
    try:
        sigmf_datatype = SIGMF_DTYPES[sample_type]
    except KeyError:
        print(f"Unknown sample type in ZST file name: {sample_type}")
        return None
        
    #print(f"Parsed ZST file: {filename} (timestamp: {timestamp}, freq_center: {freq_center}, sample_rate: {sample_rate}, sample_type: {sample_type})")
    sample_dtype, sample_type = SAMPLE_DTYPES.get(sample_type, (None, None))
    sample_bits = None
    sample_len = None
    if sample_dtype:
        sample_dtype = np.dtype([("i", sample_dtype), ("q", sample_dtype)])
        sample_bits = sample_dtype[0].itemsize * 8
        sample_len = sample_dtype[0].itemsize * 2

    file_info = {
        "filename": filename,
        "freq_center": freq_center,
        "sample_rate": sample_rate,
        "sample_dtype": sample_dtype,
        "sample_len": sample_len,  # number of bytes
        "sample_type": sample_type,
        "sample_bits": sample_bits,
        "timestamp": timestamp,
        "sigmf_datatype": sigmf_datatype,
    }
    return file_info
