
#include <vector>
#include <map>
#include <numeric>

#include "nothing.hpp"

//
// CSR to map
//
template<typename IndexT, typename NumericT, typename RowT>
void run(std::vector<IndexT> const & A_rows, std::vector<IndexT> const & A_cols, std::vector<NumericT> const & A_values,
         std::vector<RowT> & B)
{
  std::size_t N = A_rows.size() - 1;
  B.resize(N);

  IndexT row_start = A_rows[0];
  for (std::size_t row = 0; row < N; ++row)
  {
    IndexT row_stop  = A_rows[row+1];

    for (IndexT nnz_index = row_start; nnz_index < row_stop; ++nnz_index)
      B[A_cols[nnz_index]][row] = A_values[nnz_index];

    row_start = row_stop;
  }

  do_nothing(B);
}


// create outer pointers my way, for comparison
template<typename IndexT, typename NumericT>
void
stdalg_run(std::vector<IndexT> const & A_rows, std::vector<IndexT> const & A_cols, std::vector<NumericT> const & A_values,
           std::vector<IndexT>       & B_rows, std::vector<IndexT>       & B_cols, std::vector<NumericT>       & B_values)
{
    std::size_t N = A_rows.size() - 1;

    B_cols.resize(A_cols.size());
    B_values.resize(A_values.size()); // note: initialization with zero not strictly required
    B_rows = std::vector<IndexT>(A_rows.size(), 0);    // valid only for square matrices

    // 1) count the number of entries in each column (soon to be row)
    std::for_each(A_cols.begin(), A_cols.end(),
                  [&B_rows](IndexT col) { ++B_rows[col+1]; });

    // 2) perform an "exclusive sum" on the result
    std::partial_sum(B_rows.begin(), B_rows.end(), B_rows.begin());

    // 3) copy values (and move rows to columns)

    for (IndexT col_in_A = 0; col_in_A < N; ++col_in_A) {
        // this is the loop that should run in parallel
        // use counting iterator

        IndexT B_offset = B_rows[col_in_A]; // index of first unwritten element in this row

        // next phase: an iterator returning row and value pairs for a given column
        // write this custom with iterator_facade
        // finally zip iterator and run copy?  But B_offsets would have to be zipped iterators

        // look for this column in each row
        for (IndexT row = 0; row < N; ++row) {
            // take advantage of sorted column numbers to locate, if present
            auto col_range = std::equal_range(A_cols.begin() + A_rows[row],
                                              A_cols.begin() + A_rows[row+1],
                                              col_in_A);
            if (col_range.first != col_range.second) {
                // non-empty range means we have found the target value
                // Add this row (and the associated value) to B
                IndexT nnz_index = std::distance(A_cols.begin(), col_range.first);
                IndexT B_nnz_index = B_offset;
                B_cols[B_nnz_index] = row;
                B_values[B_nnz_index] = A_values[nnz_index];
                B_offset++;
            }
        }
    }
}

//
// CSR to CSR
//
template<typename IndexT, typename NumericT>
void run(std::vector<IndexT> const & A_rows, std::vector<IndexT> const & A_cols, std::vector<NumericT> const & A_values,
         std::vector<IndexT>       & B_rows, std::vector<IndexT>       & B_cols, std::vector<NumericT>       & B_values)
{
  B_cols.resize(A_cols.size());
  B_values.resize(A_values.size()); // note: initialization with zero not strictly required

  B_rows = std::vector<IndexT>(A_rows.size(), 0);
  std::size_t N = A_rows.size() - 1;

  //
  // Stage 1: Compute pattern for B
  //
  IndexT row_start = 0;
  for (std::size_t row = 0; row < N; ++row)
  {
    IndexT row_stop  = A_rows[row+1];

    for (IndexT nnz_index = row_start; nnz_index < row_stop; ++nnz_index)
      B_rows[A_cols[nnz_index]] += 1;

    row_start = row_stop;
  }

  // Bring row-start array in place using exclusive-scan:
  IndexT offset = 0;
  for (std::size_t row = 0; row < N; ++row)
  {
    IndexT tmp = B_rows[row];
    B_rows[row] = offset;
    offset += tmp;
  }
  B_rows[N] = offset;


  //
  // Stage 2: Fill with data
  //
  std::vector<IndexT> B_offsets(B_rows); // index of first unwritten element per row

  row_start = A_rows[0];
  for (std::size_t row = 0; row < N; ++row)
  {
    IndexT row_stop  = A_rows[row+1];

    for (IndexT nnz_index = row_start; nnz_index < row_stop; ++nnz_index)
    {
      IndexT col_in_A = A_cols[nnz_index];
      IndexT B_nnz_index = B_offsets[col_in_A];
      B_cols[B_nnz_index] = row;
      B_values[B_nnz_index] = A_values[nnz_index];
      B_offsets[col_in_A] += 1;
    }

    row_start = row_stop;
  }

  do_nothing(B_rows, B_cols, B_values);
}

