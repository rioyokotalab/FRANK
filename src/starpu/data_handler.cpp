#include "hicma_private/starpu_data_handler.h"

#include "hicma/classes/initialization_helpers/index_range.h"

#include <memory>
#include <vector>

//introduced in commit 9c641d29569cfd5fc9be8a185a7144f6d95f0160

namespace hicma
{

DataHandler::~DataHandler() {}

DataHandler::DataHandler(int64_t n_rows, int64_t n_cols, double val)
: data(std::make_shared<std::vector<double>>(n_rows*n_cols, val)) {}

DataHandler::DataHandler(
  std::shared_ptr<DataHandler> parent,
  std::shared_ptr<std::vector<double>> data
) : data(data), parent(parent) {}

double& DataHandler::operator[](int64_t i) { return (*data)[i]; }

const double& DataHandler::operator[](int64_t i) const { return (*data)[i]; }

uint64_t DataHandler::size() const { return data->size(); }

struct partition_args { std::vector<IndexRange> row_ranges, col_ranges; };

std::vector<std::shared_ptr<DataHandler>> DataHandler::split(
  std::shared_ptr<DataHandler> parent,
  const std::vector<IndexRange>& row_ranges,
  const std::vector<IndexRange>& col_ranges
) {
  size_t n_children = row_ranges.size()*col_ranges.size();
  std::vector<std::shared_ptr<DataHandler>> out(n_children);
  for (uint64_t i=0; i<row_ranges.size(); ++i) {
    for (uint64_t j=0; j<col_ranges.size(); ++j) {
      out[i*col_ranges.size()+j] = std::make_shared<DataHandler>(
        parent, data
      );
    }
  }
  return out;
}

bool DataHandler::is_child() const { return parent.get() != nullptr; }

} // namespace hicma
