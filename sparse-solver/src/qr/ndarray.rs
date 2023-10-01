use crate::qr::SparseQRError;

use super::{DenseMatrix, DenseMatrixView, DenseMatrixViewMut, Layout, SparseQRResult};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2};

impl DenseMatrix for Array1<f64> {
    type View<'a> = ArrayView1<'a, f64>;
    type ViewMut<'a> = ArrayViewMut1<'a, f64>;
    fn view(&self) -> Self::View<'_> {
        self.view()
    }
    fn view_mut(&mut self) -> Self::ViewMut<'_> {
        self.view_mut()
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
        let ld = self.strides()[0];
        Ok(ld as usize)
    }
    fn values(&self) -> &[f64] {
        self.as_slice().unwrap()
    }
    fn values_mut(&mut self) -> &mut [f64] {
        self.as_slice_mut().unwrap()
    }
}

impl DenseMatrixView for ArrayView1<'_, f64> {
    type Owned = Array1<f64>;
    fn to_owned(&self) -> Self::Owned {
        self.to_owned()
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
        let ld = self.strides()[0];
        Ok(ld as usize)
    }
    fn values(&self) -> &[f64] {
        self.as_slice().unwrap()
    }
}

impl DenseMatrixView for ArrayViewMut1<'_, f64> {
    type Owned = Array1<f64>;
    fn to_owned(&self) -> Self::Owned {
        self.to_owned()
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
        let ld = self.strides()[0];
        Ok(ld as usize)
    }
    fn values(&self) -> &[f64] {
        self.as_slice().unwrap()
    }
}

impl DenseMatrixViewMut for ArrayViewMut1<'_, f64> {
    fn values_mut(&mut self) -> &mut [f64] {
        self.as_slice_mut().unwrap()
    }
}

impl DenseMatrix for Array2<f64> {
    type View<'a> = ArrayView2<'a, f64>;
    type ViewMut<'a> = ArrayViewMut2<'a, f64>;
    fn view(&self) -> Self::View<'_> {
        self.view()
    }
    fn view_mut(&mut self) -> Self::ViewMut<'_> {
        self.view_mut()
    }
    fn n_rows(&self) -> usize {
        self.nrows()
    }
    fn n_cols(&self) -> usize {
        self.ncols()
    }
    fn layout(&self) -> SparseQRResult<Layout> {
        let ld = self.strides();
        if ld[0] == 1 {
            Ok(Layout::ColumnMajor)
        } else if ld[1] == 1 {
            Ok(Layout::RowMajor)
        } else {
            Err(SparseQRError::NotSupported(
                "Non-contiguous array".to_string(),
            ))
        }
    }
    fn leading_dim(&self) -> SparseQRResult<usize> {
        let ld = self.strides();
        if ld[0] == 1 {
            Ok(ld[1] as usize)
        } else if ld[1] == 1 {
            Ok(ld[0] as usize)
        } else {
            Err(SparseQRError::NotSupported(
                "Non-contiguous array".to_string(),
            ))
        }
    }
    fn values(&self) -> &[f64] {
        self.as_slice().unwrap()
    }
    fn values_mut(&mut self) -> &mut [f64] {
        self.as_slice_mut().unwrap()
    }
}

impl DenseMatrixView for ArrayView2<'_, f64> {
    type Owned = Array2<f64>;
    fn to_owned(&self) -> Self::Owned {
        self.to_owned()
    }
    fn n_rows(&self) -> usize {
        self.nrows()
    }
    fn n_cols(&self) -> usize {
        self.ncols()
    }
    fn layout(&self) -> SparseQRResult<Layout> {
        let ld = self.strides();
        if ld[0] == 1 {
            Ok(Layout::ColumnMajor)
        } else if ld[1] == 1 {
            Ok(Layout::RowMajor)
        } else {
            Err(SparseQRError::NotSupported(
                "Non-contiguous array".to_string(),
            ))
        }
    }
    fn leading_dim(&self) -> SparseQRResult<usize> {
        let ld = self.strides();
        if ld[0] == 1 {
            Ok(ld[1] as usize)
        } else if ld[1] == 1 {
            Ok(ld[0] as usize)
        } else {
            Err(SparseQRError::NotSupported(
                "Non-contiguous array".to_string(),
            ))
        }
    }
    fn values(&self) -> &[f64] {
        self.as_slice().unwrap()
    }
}

impl DenseMatrixView for ArrayViewMut2<'_, f64> {
    type Owned = Array2<f64>;
    fn to_owned(&self) -> Self::Owned {
        self.to_owned()
    }
    fn n_rows(&self) -> usize {
        self.nrows()
    }
    fn n_cols(&self) -> usize {
        self.ncols()
    }
    fn layout(&self) -> SparseQRResult<Layout> {
        let ld = self.strides();
        if ld[0] == 1 {
            Ok(Layout::ColumnMajor)
        } else if ld[1] == 1 {
            Ok(Layout::RowMajor)
        } else {
            Err(SparseQRError::NotSupported(
                "Non-contiguous array".to_string(),
            ))
        }
    }
    fn leading_dim(&self) -> SparseQRResult<usize> {
        let ld = self.strides();
        if ld[0] == 1 {
            Ok(ld[1] as usize)
        } else if ld[1] == 1 {
            Ok(ld[0] as usize)
        } else {
            Err(SparseQRError::NotSupported(
                "Non-contiguous array".to_string(),
            ))
        }
    }
    fn values(&self) -> &[f64] {
        self.as_slice().unwrap()
    }
}

impl DenseMatrixViewMut for ArrayViewMut2<'_, f64> {
    fn values_mut(&mut self) -> &mut [f64] {
        self.as_slice_mut().unwrap()
    }
}
