/**
 * @file bindings/cli/get_printable_param.hpp
 * @author Ryan Curtin
 *
 * Print the parameter to stdout, using template metaprogramming to enforce
 * different behavior for different parameter types.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_CLI_GET_PRINTABLE_PARAM_HPP
#define MLPACK_BINDINGS_CLI_GET_PRINTABLE_PARAM_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/core/util/param_data.hpp>
#include <mlpack/core/util/is_std_vector.hpp>

namespace mlpack {
namespace bindings {
namespace cli {

/**
 * Print an option.
 */
template<typename T>
std::string GetPrintableParam(
    util::ParamData& data,
    const std::enable_if_t<!arma::is_arma_type<T>::value>* = 0,
    const std::enable_if_t<!util::IsStdVector<T>::value>* = 0,
    const std::enable_if_t<!data::HasSerialize<T>::value>* = 0,
    const std::enable_if_t<!std::is_same_v<T,
        std::tuple<data::DatasetInfo, arma::mat>>>* = 0);

/**
 * Print a vector option, with spaces between it.
 */
template<typename T>
std::string GetPrintableParam(
    util::ParamData& data,
    const std::enable_if_t<util::IsStdVector<T>::value>* = 0);

/**
 * Print a matrix/tuple option (this just prints the filename).
 */
template<typename T>
std::string GetPrintableParam(
    util::ParamData& data,
    const std::enable_if_t<arma::is_arma_type<T>::value || std::is_same_v<T,
        std::tuple<data::DatasetInfo, arma::mat>>>* = 0);

/**
 * Print a model option (this just prints the filename).
 */
template<typename T>
std::string GetPrintableParam(
    util::ParamData& data,
    const std::enable_if_t<!arma::is_arma_type<T>::value>* = 0,
    const std::enable_if_t<data::HasSerialize<T>::value>* = 0);

/**
 * Print an option into a std::string.  This should print a short, one-line
 * representation of the object.  The string will be stored in the output
 * pointer.
 */
template<typename T>
void GetPrintableParam(util::ParamData& data,
                       const void* /* input */,
                       void* output)
{
  *((std::string*) output) =
      GetPrintableParam<std::remove_pointer_t<T>>(data);
}

} // namespace cli
} // namespace bindings
} // namespace mlpack

// Include implementation.
#include "get_printable_param_impl.hpp"

#endif
