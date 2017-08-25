#include <benchmark/benchmark.h>

#include <vector>
#include <map>
#include <numeric>

#include <range/v3/algorithm/for_each.hpp>
#include <range/v3/algorithm/sort.hpp>
#include <range/v3/view/concat.hpp>
#include <range/v3/view/single.hpp>
#include <range/v3/view/take.hpp>
#include <range/v3/view/iota.hpp>
#include <range/v3/view/tail.hpp>
#include <range/v3/view/repeat.hpp>
#include <range/v3/view/repeat_n.hpp>
#include <range/v3/view/for_each.hpp>
#include <range/v3/view/slice.hpp>
#include <range/v3/view/zip.hpp>
#include <range/v3/view/empty.hpp>
#include <range/v3/view/any_view.hpp>

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

  benchmark::DoNotOptimize(B);
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
  std::for_each(A_cols.begin(), A_cols.end(),
                [&B_rows](IndexT col) { ++B_rows[col+1]; });

  // Bring row-start array in place using exclusive-scan:
  std::partial_sum(B_rows.begin(), B_rows.end(), B_rows.begin());

  //
  // Stage 2: Fill with data
  //
  std::vector<IndexT> B_offsets(B_rows); // index of first unwritten element per row

  IndexT row_start = A_rows[0];
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

  benchmark::DoNotOptimize(B_rows);
  benchmark::DoNotOptimize(B_cols);
  benchmark::DoNotOptimize(B_values);
}

//
// CSR to CSR via range-v3
//

// utility view: adjacent_difference
// range-v3 only provides it as a mutating algorithm

template<typename Rng,
         typename Fn = std::minus<ranges::v3::range_value_type_t<Rng>>>
ranges::v3::any_view<ranges::v3::range_value_type_t<Rng>>
adj_diff_view(Rng r, Fn f = Fn()) {
  namespace rng = ranges::v3;
  using namespace rng::view;

  if (rng::empty(r)) {
    return empty<rng::range_value_type_t<Rng>>();
  }

  return concat(take(r, 1),    // first element is unchanged
                // remainder is f() applied to adjacent elements
                  zip_with(f, tail(r), slice(r, 0, rng::end-1)));
}

template<typename IndexT, typename NumericT>
void run_range_v3(std::vector<IndexT> const & A_rows, std::vector<IndexT>         A_cols, std::vector<NumericT>         A_values,
                  std::vector<IndexT>       & B_rows, std::vector<IndexT>       & B_cols, std::vector<NumericT>       & B_values)
{
  namespace rng = ranges::v3;
  using namespace rng::view;
  using namespace std;


  // create a range holding the current row indices (future column indices) of each value

  // first get the number of entries in each row
  auto nz_row_counts = adj_diff_view(tail(A_rows));

  vector<IndexT> row_ind;
  // iterate over (row number, entry count) for each row
  row_ind = join(zip_with([](auto rowno, auto count) { return repeat_n(rowno, count); },
                          rng::view::iota(IndexT{0}),
                          nz_row_counts));

  // sort a zip of (old row indices, old column indices, values)
  // by (old column indices, old row indices) to produce a column major order

  auto col_major_zip = zip(A_cols, row_ind, A_values);

  // stable_sort using just (old column indices) will also work here - need to investigate perf
  rng::sort(col_major_zip,
            [](auto a, auto b) { return tie(get<0>(a), get<1>(a)) < tie(get<0>(b), get<1>(b)); });

  // swap the sorted row indices into place as the new columns
  swap(A_cols, row_ind);
  swap(B_cols, A_cols);
  swap(B_values, A_values);

  // scan the new row indices to locate row boundaries
  auto row_ind_it = begin(row_ind);
  IndexT old_row_cnt = A_rows.size() - 1;
  B_rows = rng::view::for_each(rng::view::iota(IndexT{0}, old_row_cnt + 1),
                               [&](IndexT row) {
                                 row_ind_it = lower_bound(row_ind_it, row_ind.end(), row);
                                 return single(distance(row_ind.begin(), row_ind_it));
                               });

  benchmark::DoNotOptimize(B_rows);
  benchmark::DoNotOptimize(B_cols);
  benchmark::DoNotOptimize(B_values);
}
