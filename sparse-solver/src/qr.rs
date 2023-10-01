use std::borrow::Cow;
use thiserror::Error;

#[derive(Error, Debug, PartialEq, Eq)]
pub enum SparseQRError {
    #[error("The routine encountered an empty handle or matrix array")]
    NotInitialized,
    #[error("Internal memory allocation failed")]
    AllocFailed,
    #[error("The input parameters contain an invalid value")]
    InvalidValue,
    #[error("Execution failed")]
    ExecutionFailed,
    #[error("An error in algorithm implementation occurred")]
    InternalError,
    #[error("The requested operation is not supported")]
    NotSupported(String),
}

type SparseQRResult<T> = Result<T, SparseQRError>;

pub enum Layout {
    RowMajor,
    ColumnMajor,
}

pub trait CSRMatrix {
    fn n_rows(&self) -> usize;
    fn n_cols(&self) -> usize;
    fn rows_pointer(&self) -> Cow<'_, [i32]>;
    fn col_index(&self) -> Cow<'_, [i32]>;
    fn values(&self) -> Cow<'_, [f64]>;
}

pub trait DenseMatrix {
    type View<'a>: DenseMatrixView
    where
        Self: 'a;
    type ViewMut<'a>: DenseMatrixView + DenseMatrixViewMut
    where
        Self: 'a;
    fn view<'a>(&'a self) -> Self::View<'a>;
    fn view_mut<'a>(&'a mut self) -> Self::ViewMut<'a>;
    fn n_rows(&self) -> usize;
    fn n_cols(&self) -> usize;
    fn layout(&self) -> SparseQRResult<Layout>;
    fn leading_dim(&self) -> SparseQRResult<usize>;
    fn values(&self) -> &[f64];
    fn values_mut(&mut self) -> &mut [f64];
}

pub trait DenseMatrixView {
    type Owned: DenseMatrix;
    fn to_owned(&self) -> Self::Owned;
    fn n_rows(&self) -> usize;
    fn n_cols(&self) -> usize;
    fn layout(&self) -> SparseQRResult<Layout>;
    fn leading_dim(&self) -> SparseQRResult<usize>;
    fn values(&self) -> &[f64];
}

pub trait DenseMatrixViewMut {
    fn values_mut(&mut self) -> &mut [f64];
}

pub mod ndarray;
pub mod sprs;
pub mod vec;

pub use interface::{solve, solve_inplace, solve_into};

pub mod interface {
    use super::*;
    use onemkl_sparse_solver_sys::sparse_qr::*;

    impl From<u32> for SparseQRError {
        fn from(status: u32) -> Self {
            match status {
                sparse_status_t_SPARSE_STATUS_NOT_INITIALIZED => SparseQRError::NotInitialized,
                sparse_status_t_SPARSE_STATUS_ALLOC_FAILED => SparseQRError::AllocFailed,
                sparse_status_t_SPARSE_STATUS_INVALID_VALUE => SparseQRError::InvalidValue,
                sparse_status_t_SPARSE_STATUS_EXECUTION_FAILED => SparseQRError::ExecutionFailed,
                sparse_status_t_SPARSE_STATUS_INTERNAL_ERROR => SparseQRError::InternalError,
                sparse_status_t_SPARSE_STATUS_NOT_SUPPORTED => {
                    SparseQRError::NotSupported("".to_string())
                }
                _ => panic!("Unknown status {}", status),
            }
        }
    }

    pub fn solve<M: DenseMatrixView>(a: &impl CSRMatrix, b: M) -> SparseQRResult<M::Owned> {
        let mut x = b.to_owned();
        solve_into(a, b, x.view_mut())?;
        Ok(x)
    }

    pub fn solve_into(
        a: &impl CSRMatrix,
        b: impl DenseMatrixView,
        mut x: impl DenseMatrixViewMut + DenseMatrixView,
    ) -> SparseQRResult<()> {
        let csr_a = {
            let n_rows = a.n_rows();
            let n_cols = a.n_cols();
            let rows_start = a.rows_pointer();
            let rows_end = &rows_start[1..];
            let col_index = a.col_index();
            let values = a.values();

            let mut csr_a = std::ptr::null_mut();
            let status = unsafe {
                mkl_sparse_d_create_csr(
                    &mut csr_a,
                    sparse_index_base_t_SPARSE_INDEX_BASE_ZERO,
                    n_rows as i32,
                    n_cols as i32,
                    rows_start.as_ptr() as *mut i32,
                    rows_end.as_ptr() as *mut i32,
                    col_index.as_ptr() as *mut i32,
                    values.as_ptr() as *mut f64,
                )
            };
            if status != sparse_status_t_SPARSE_STATUS_SUCCESS {
                //panic!("Failed to create CSR matrix: {}", status);
                return Err(status.into());
            }
            csr_a
        };

        let descr_a = matrix_descr {
            type_: sparse_matrix_type_t_SPARSE_MATRIX_TYPE_GENERAL,
            mode: 0,
            diag: 0,
        };

        let layout = match b.layout()? {
            Layout::RowMajor => sparse_layout_t_SPARSE_LAYOUT_ROW_MAJOR,
            Layout::ColumnMajor => sparse_layout_t_SPARSE_LAYOUT_COLUMN_MAJOR,
        };

        let status = unsafe {
            mkl_sparse_d_qr(
                sparse_operation_t_SPARSE_OPERATION_NON_TRANSPOSE,
                csr_a,
                descr_a,
                layout,
                b.n_cols() as i32,
                x.values_mut().as_mut_ptr(),
                x.leading_dim()? as i32,
                b.values().as_ptr(),
                b.leading_dim()? as i32,
            )
        };
        if status != sparse_status_t_SPARSE_STATUS_SUCCESS {
            //panic!("Failed to solve linear system: {}", status);
            return Err(status.into());
        }

        let status = unsafe { mkl_sparse_destroy(csr_a) };
        if status != sparse_status_t_SPARSE_STATUS_SUCCESS {
            //panic!("Failed to destroy CSR matrix: {}", status);
            return Err(status.into());
        }

        Ok(())
    }

    pub fn solve_inplace<M: DenseMatrix>(a: &impl CSRMatrix, b: M) -> SparseQRResult<M> {
        let csr_a = {
            let n_rows = a.n_rows();
            let n_cols = a.n_cols();
            let rows_start = a.rows_pointer();
            let rows_end = &rows_start[1..];
            let col_index = a.col_index();
            let values = a.values();

            let mut csr_a = std::ptr::null_mut();
            let status = unsafe {
                mkl_sparse_d_create_csr(
                    &mut csr_a,
                    sparse_index_base_t_SPARSE_INDEX_BASE_ZERO,
                    n_rows as i32,
                    n_cols as i32,
                    rows_start.as_ptr() as *mut i32,
                    rows_end.as_ptr() as *mut i32,
                    col_index.as_ptr() as *mut i32,
                    values.as_ptr() as *mut f64,
                )
            };
            if status != sparse_status_t_SPARSE_STATUS_SUCCESS {
                return Err(status.into());
            }
            csr_a
        };

        let descr_a = matrix_descr {
            type_: sparse_matrix_type_t_SPARSE_MATRIX_TYPE_GENERAL,
            mode: 0,
            diag: 0,
        };

        let layout = match b.layout()? {
            Layout::RowMajor => sparse_layout_t_SPARSE_LAYOUT_ROW_MAJOR,
            Layout::ColumnMajor => sparse_layout_t_SPARSE_LAYOUT_COLUMN_MAJOR,
        };

        let status = unsafe {
            mkl_sparse_d_qr(
                sparse_operation_t_SPARSE_OPERATION_NON_TRANSPOSE,
                csr_a,
                descr_a,
                layout,
                b.n_cols() as i32,
                b.values().as_ptr() as *mut f64,
                b.leading_dim()? as i32,
                b.values().as_ptr(),
                b.leading_dim()? as i32,
            )
        };
        if status != sparse_status_t_SPARSE_STATUS_SUCCESS {
            return Err(status.into());
        }

        let status = unsafe { mkl_sparse_destroy(csr_a) };
        if status != sparse_status_t_SPARSE_STATUS_SUCCESS {
            return Err(status.into());
        }

        Ok(b)
    }
}

#[cfg(test)]
mod tests {
    use super::{solve, solve_inplace, solve_into};
    use sprs::{CsMatI, TriMatI};

    #[test]
    fn test_solver() {
        let mut a = TriMatI::new((4, 4));
        a.add_triplet(0, 0, -2.0);
        a.add_triplet(0, 1, 1.0);
        a.add_triplet(1, 0, 1.0);
        a.add_triplet(1, 1, -2.0);
        a.add_triplet(1, 2, 1.0);
        a.add_triplet(2, 1, 1.0);
        a.add_triplet(2, 2, -2.0);
        a.add_triplet(2, 3, 1.0);
        a.add_triplet(3, 2, 1.0);
        a.add_triplet(3, 3, -2.0);
        let a: CsMatI<_, _> = a.to_csr::<i32>();

        let b = vec![1.0, 0.0, 0.0, 0.0];
        let mut x = vec![0.; b.len()];

        solve_into(&a, &b[..], &mut x[..]).unwrap();
        let x2 = solve(&a, &b[..]).unwrap();
        let x3 = solve_inplace(&a, b).unwrap();

        for ((&v1, &v2), &v3) in x.iter().zip(&x2).zip(&x3) {
            assert_eq!(v1, v2);
            assert_eq!(v1, v3);
        }
    }
}
