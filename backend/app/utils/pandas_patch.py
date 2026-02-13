import pandas as pd
import logging

logger = logging.getLogger(__name__)


def patch_pandas_append():
    """
    Patch pd.Series.append/pd.DataFrame.append and _append which were removed in pandas 2.0.
    This allows legacy libraries (like pandas-ta) to continue working.
    """
    try:
        # Patch Series.append and _append
        if not hasattr(pd.Series, "append"):

            def append(self, to_append, ignore_index=False, verify_integrity=False):
                from pandas import concat

                if isinstance(to_append, (list, tuple)):
                    to_concat = [self] + list(to_append)
                else:
                    to_concat = [self, to_append]
                return concat(
                    to_concat,
                    ignore_index=ignore_index,
                    verify_integrity=verify_integrity,
                )

            pd.Series.append = append

        if not hasattr(pd.Series, "_append"):
            # Some libraries might access the private method _append
            pd.Series._append = pd.Series.append

        # Patch DataFrame.append and _append
        if not hasattr(pd.DataFrame, "append"):

            def append(
                self, other, ignore_index=False, verify_integrity=False, sort=False
            ):
                from pandas import concat

                return concat(
                    [self, other],
                    ignore_index=ignore_index,
                    verify_integrity=verify_integrity,
                    sort=sort,
                )

            pd.DataFrame.append = append

        if not hasattr(pd.DataFrame, "_append"):
            pd.DataFrame._append = pd.DataFrame.append

        logger.info("Applied pandas append/_append patch for compatibility.")

    except Exception as e:
        logger.warning(f"Failed to patch pandas append: {e}")
