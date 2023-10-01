use super::{DenseMatrix, DenseMatrixView, DenseMatrixViewMut, Layout, SparseQRResult};

impl DenseMatrix for Vec<f64> {
    type View<'a> = &'a [f64];
    type ViewMut<'a> = &'a mut [f64];
    fn view(&self) -> Self::View<'_> {
        self.as_slice()
    }
    fn view_mut<'a>(&'a mut self) -> Self::ViewMut<'a> {
        self.as_mut_slice()
    }
    fn n_rows(&self) -> usize {
        self.len()
    }
    fn n_cols(&self) -> usize {
        1
    }
    fn layout(&self) -> SparseQRResult<Layout> {
        Ok(Layout::RowMajor)
    }
    fn leading_dim(&self) -> SparseQRResult<usize> {
        Ok(1)
    }
    fn values(&self) -> &[f64] {
        self.as_slice()
    }
    fn values_mut(&mut self) -> &mut [f64] {
        self.as_mut_slice()
    }
}

impl DenseMatrixView for &[f64] {
    type Owned = Vec<f64>;
    fn to_owned(&self) -> Self::Owned {
        self.to_vec()
    }
    fn n_rows(&self) -> usize {
        self.len()
    }
    fn n_cols(&self) -> usize {
        1
    }
    fn layout(&self) -> SparseQRResult<Layout> {
        Ok(Layout::RowMajor)
    }
    fn leading_dim(&self) -> SparseQRResult<usize> {
        Ok(1)
    }
    fn values(&self) -> &[f64] {
        self
    }
}

impl DenseMatrixView for &mut [f64] {
    type Owned = Vec<f64>;
    fn to_owned(&self) -> Self::Owned {
        self.to_vec()
    }
    fn n_rows(&self) -> usize {
        self.len()
    }
    fn n_cols(&self) -> usize {
        1
    }
    fn layout(&self) -> SparseQRResult<Layout> {
        Ok(Layout::RowMajor)
    }
    fn leading_dim(&self) -> SparseQRResult<usize> {
        Ok(1)
    }
    fn values(&self) -> &[f64] {
        self
    }
}

impl DenseMatrixViewMut for &mut [f64] {
    fn values_mut(&mut self) -> &mut [f64] {
        self
    }
}
