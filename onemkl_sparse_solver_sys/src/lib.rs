pub mod pardiso;
pub mod rci;
pub mod sparse_qr;

extern crate intel_mkl_src;

#[cfg(test)]
mod tests {
    #[test]
    fn sparse_qr() {
        use super::sparse_qr::*;

        let rows = 4;
        let cols = 4;
        let mut rows_start = vec![0, 2, 5, 8, 10];
        let mut rows_end = (&rows_start[1..]).to_vec();
        let mut col_index = vec![0, 1, 0, 1, 2, 1, 2, 3, 2, 3];
        let mut values = vec![-2., 1., 1., -2., 1., 1., -2., 1., 1., -2.];
        assert_eq!(col_index.len(), values.len());
        assert_eq!(col_index.len() as i32, rows_end[rows_end.len() - 1]);

        let mut csr_a = std::ptr::null_mut();
        let status = unsafe {
            mkl_sparse_d_create_csr(
                &mut csr_a,
                sparse_index_base_t_SPARSE_INDEX_BASE_ZERO,
                rows,
                cols,
                rows_start.as_mut_ptr(),
                rows_end.as_mut_ptr(),
                col_index.as_mut_ptr(),
                values.as_mut_ptr(),
            )
        };
        if status != sparse_status_t_SPARSE_STATUS_SUCCESS {
            panic!("mkl_sparse_d_create_csr failed with status {}", status);
        }

        let descr_a = matrix_descr {
            type_: sparse_matrix_type_t_SPARSE_MATRIX_TYPE_GENERAL,
            mode: 0,
            diag: 0,
        };

        let status = unsafe { mkl_sparse_qr_reorder(csr_a, descr_a) };
        if status != sparse_status_t_SPARSE_STATUS_SUCCESS {
            panic!("mkl_sparse_qr_reorder failed with status {}", status);
        }

        let alt_values = std::ptr::null_mut();

        let status = unsafe { mkl_sparse_d_qr_factorize(csr_a, alt_values) };
        if status != sparse_status_t_SPARSE_STATUS_SUCCESS {
            panic!("mkl_sparse_d_qr_factorize failed with status {}", status);
        }

        let mut b = vec![1., 0., 0., 0.];
        let mut x = vec![0.; b.len()];

        let status = unsafe {
            mkl_sparse_d_qr_qmult(
                sparse_operation_t_SPARSE_OPERATION_NON_TRANSPOSE,
                csr_a,
                sparse_layout_t_SPARSE_LAYOUT_ROW_MAJOR,
                1,
                x.as_mut_ptr(),
                1,
                b.as_mut_ptr(),
                1,
            )
        };
        if status != sparse_status_t_SPARSE_STATUS_SUCCESS {
            panic!("mkl_sparse_d_qr_qmult failed with status {}", status);
        }

        let status = unsafe {
            mkl_sparse_d_qr_rsolve(
                sparse_operation_t_SPARSE_OPERATION_NON_TRANSPOSE,
                csr_a,
                sparse_layout_t_SPARSE_LAYOUT_ROW_MAJOR,
                1,
                x.as_mut_ptr(),
                1,
                x.as_mut_ptr(),
                1,
            )
        };
        if status != sparse_status_t_SPARSE_STATUS_SUCCESS {
            panic!("mkl_sparse_d_qr_rsolve failed with status {}", status);
        }

        let status = unsafe { mkl_sparse_destroy(csr_a) };
        if status != sparse_status_t_SPARSE_STATUS_SUCCESS {
            panic!("mkl_sparse_destroy failed with status {}", status);
        }
    }
}
