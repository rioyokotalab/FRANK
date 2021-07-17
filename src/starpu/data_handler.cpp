#include "hicma_private/starpu_data_handler.h"

#include "hicma/classes/initialization_helpers/index_range.h"

#include "starpu.h"

#include <memory>
#include <vector>


namespace hicma
{

DataHandler::~DataHandler() {
  if (starpu_is_initialized()) {
    if (splits.size() > 0) {
      for (std::vector<starpu_data_handle_t> child_handles : splits) {
        starpu_data_partition_clean(
          handle, child_handles.size(), &child_handles[0]
        );
      }
    }
    if (!is_child()) {
      starpu_data_unregister_submit(handle);
    }
  }
}

DataHandler::DataHandler(int64_t n_rows, int64_t n_cols, float val)
: data(std::make_shared<std::vector<float>>(n_rows*n_cols, val))
{
  if (starpu_is_initialized()) {
    starpu_matrix_data_register(
      &handle, STARPU_MAIN_RAM,
      (uintptr_t)&(*data)[0], n_cols, n_cols, n_rows, sizeof((*data)[0])
    );
  }
}

DataHandler::DataHandler(
  std::shared_ptr<DataHandler> parent,
  std::shared_ptr<std::vector<float>> data,
  starpu_data_handle_t handle
) : data(data), parent(parent), handle(handle) {}

float& DataHandler::operator[](int64_t i) { return (*data)[i]; }

const float& DataHandler::operator[](int64_t i) const { return (*data)[i]; }

uint64_t DataHandler::size() const { return data->size(); }

struct partition_args { std::vector<IndexRange> row_ranges, col_ranges; };

void partition_filter(
  void* father_interface, void* child_interface,
  starpu_data_filter* filter, unsigned id, unsigned
) {
	starpu_matrix_interface* father = (starpu_matrix_interface*) father_interface;
	starpu_matrix_interface* child = (starpu_matrix_interface*) child_interface;
  partition_args* args = (partition_args*) filter->filter_arg_ptr;

  unsigned i = id / args->col_ranges.size();
  unsigned j = id % args->col_ranges.size();
  child->id = father->id;
  child->ny = args->row_ranges[i].n;
  child->nx = args->col_ranges[j].n;
  child->elemsize = father->elemsize;
  child->allocsize = child->nx * child->ny * child->elemsize;

  size_t offset = (
    args->row_ranges[i].start*father->ld+args->col_ranges[j].start
  ) * father->elemsize;

	if (father->dev_handle) {
		if (father->ptr) child->ptr = father->ptr + offset;
		child->ld = father->ld;
		child->dev_handle = father->dev_handle;
		child->offset = father->offset + offset;
	}
}

std::vector<std::shared_ptr<DataHandler>> DataHandler::split(
  std::shared_ptr<DataHandler> parent,
  const std::vector<IndexRange>& row_ranges,
  const std::vector<IndexRange>& col_ranges
) {
  size_t n_children = row_ranges.size()*col_ranges.size();
  std::vector<std::shared_ptr<DataHandler>> out(n_children);
  std::vector<starpu_data_handle_t> child_handles(n_children);
  if (starpu_is_initialized()) {
    partition_args args{row_ranges, col_ranges};
    starpu_data_filter filter;
    filter.nchildren = row_ranges.size() * col_ranges.size();
    filter.filter_func = partition_filter;
    filter.filter_arg_ptr = &args;
    filter.get_child_ops = NULL;
    filter.get_nchildren = NULL;
    starpu_data_partition_plan(
      handle, &filter, &child_handles[0]
    );
  }
  for (uint64_t i=0; i<row_ranges.size(); ++i) {
    for (uint64_t j=0; j<col_ranges.size(); ++j) {
      out[i*col_ranges.size()+j] = std::make_shared<DataHandler>(
        parent, data, child_handles[i*col_ranges.size()+j]
      );
    }
  }
  splits.push_back(child_handles);
  return out;
}

starpu_data_handle_t DataHandler::get_handle() const {
  return handle;
}

bool DataHandler::is_child() const { return parent.get() != nullptr; }

} // namespace hicma
