
#include <vector>
#include <map>
#include <numeric>

#include <boost/iterator/zip_iterator.hpp>
#include <boost/fusion/adapted/std_pair.hpp>

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


template<typename IndexT, typename NumericT>
struct colwise_nz_iterator : boost::iterator_facade<colwise_nz_iterator<IndexT, NumericT>,
                                                    std::pair<IndexT, NumericT>,
                                                    std::forward_iterator_tag,
                                                    std::pair<IndexT, NumericT> const&>
{
    colwise_nz_iterator(std::vector<IndexT> const&   rows,
                        std::vector<IndexT> const&   cols,
                        std::vector<NumericT> const& values,
                        IndexT                       col)
        : sentinel_(false), rows_(&rows), cols_(&cols), values_(&values), col_(col), row_(0)
    {
        // if there is no row 0, go forward through the rows until you find one
        // or you reach the end of the column
        advance_to_valid();
    }

    colwise_nz_iterator() : sentinel_(true) {}    // end of column reached

    // iterator_facade requirements

    std::pair<IndexT, NumericT> const & dereference() const
    {
        return it_value_;
    }

    bool equal(colwise_nz_iterator const & other) const
    {
        return ((sentinel_ && other.sentinel_) ||
                ((!sentinel_ && !other.sentinel_) &&
                 (row_ == other.row_)));
    }

    void increment() {
        ++row_;
        advance_to_valid();
    }

private:

    void advance_to_valid() {
        IndexT N = rows_->size() - 1;

        auto col_range = std::equal_range(cols_->begin() + (*rows_)[row_],
                                          cols_->begin() + (*rows_)[row_+1],
                                          col_);
        while ((col_range.first == col_range.second) && (row_ < N))
        {
            ++row_;
            col_range = std::equal_range(cols_->begin() + (*rows_)[row_],
                                         cols_->begin() + (*rows_)[row_+1],
                                         col_);
        }
        if (row_ >= N)
        {
            sentinel_ = true;
        } else {
            IndexT nnz_index = std::distance(cols_->begin(), col_range.first);
            it_value_ = std::make_pair(row_, (*values_)[nnz_index]);
        }
    }

    bool                          sentinel_;      // end sentinel.  Could use (row >= N) also?
    std::vector<IndexT> const *   rows_;
    std::vector<IndexT> const *   cols_;
    std::vector<NumericT> const * values_;
    IndexT                        col_;           // column we are traversing
    IndexT                        row_;           // the current row

    std::pair<IndexT, NumericT>   it_value_;      // for returning when dereferenced

};

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
        // construct an output iterator for the {row, value} pairs
        auto entry_it = boost::make_zip_iterator(std::make_pair(B_cols.begin() + B_offset,
                                                                B_values.begin() + B_offset));

        // now the input iterator
        auto beg = colwise_nz_iterator<IndexT, NumericT>(A_rows, A_cols, A_values, col_in_A);
        auto end = colwise_nz_iterator<IndexT, NumericT>();

        // and copy the data for this column
        std::copy(beg, end, entry_it);

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

