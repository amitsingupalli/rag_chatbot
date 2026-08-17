"""NLTK Hardlink & Security Bypass for Streamlit Cloud (Fixes CWE-59 / st_nlink=2 Security Violation)."""

from __future__ import annotations

import os
import shutil
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def apply_nltk_security_patch() -> None:
    try:
        import nltk
        
        target = Path("/tmp/nltk_data") if os.name != "nt" else Path("./scratch/nltk_data")
        target.mkdir(parents=True, exist_ok=True)
        resolved_target = str(target.resolve())
        
        os.environ["NLTK_DATA"] = resolved_target
        
        if resolved_target not in nltk.data.path:
            nltk.data.path.insert(0, resolved_target)
            
        for resource in ["stopwords", "punkt", "punkt_tab"]:
            try:
                nltk.download(resource, download_dir=resolved_target, quiet=True)
            except Exception as d_err:
                logger.debug("NLTK download %s notice: %s", resource, d_err)
                
        # Copy any llama_index static nltk_cache files to /tmp/nltk_data using copyfile (strips st_nlink>1 hardlinks)
        try:
            import llama_index.core
            static_nltk = Path(llama_index.core.__file__).parent / "_static" / "nltk_cache"
            if static_nltk.exists():
                for root, _, files in os.walk(static_nltk):
                    rel = Path(root).relative_to(static_nltk)
                    dest_dir = target / rel
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    for f in files:
                        src_f = Path(root) / f
                        dest_f = dest_dir / f
                        if not dest_f.exists():
                            try:
                                shutil.copyfile(src_f, dest_f)
                            except Exception:
                                pass
        except Exception:
            pass
            
        # Sanitize nltk.data.path to remove any hardlinked _static/nltk_cache directories
        nltk.data.path = [p for p in nltk.data.path if "_static" not in str(p)]
        logger.info("Applied NLTK Security Patch (NLTK_DATA=%s)", resolved_target)
    except Exception as exc:
        logger.warning("Failed to apply NLTK security patch: %s", exc)

apply_nltk_security_patch()
