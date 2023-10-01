use std::borrow::Cow;

use super::CSRMatrix;
use sprs;

impl CSRMatrix for sprs::CsMatI<f64, i32> {
    fn n_rows(&self) -> usize {
        self.rows()
    }

    fn n_cols(&self) -> usize {
        self.cols()
    }

    fn rows_pointer(&self) -> Cow<'_, [i32]> {
        self.proper_indptr()
    }

    fn col_index(&self) -> Cow<'_, [i32]> {
        self.indices().into()
    }

    fn values(&self) -> Cow<'_, [f64]> {
        self.data().into()
    }
}
