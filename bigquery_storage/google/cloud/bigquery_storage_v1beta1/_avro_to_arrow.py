# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Helpers to parse ReadRowsResponse messages into Arrow Tables."""

# TODO: optional imports
import numpy
import numba
import pandas
import pyarrow

# Scalar types to support.
# INT64
# FLOAT64
# BOOL
#
# Later:
# NUMERIC (decimal) ??? how is this actually serialized. Let's wait.
# DATE
# TIME
# TIMESTAMP
#
# Even later:
# DATETIME - need to parse from string


def generate_avro_to_arrow_parser(avro_schema):
    """Return a parser that takes a ReadRowsResponse message and returns a
    :class:`pyarrow.Table` object.

    Args:
        avro_schema (Map):
            Avro schema in JSON format.

    Returns:
        A function that takes a message and returns a table.
    """
    message_to_buffers = numba.jit(nopython=True, nogil=True)(_avro_df)
    def message_to_table(message):
        row_count = message.avro_rows.row_count
        block = message.avro_rows.serialized_binary_rows
        int_col, float_col, bool_col = message_to_buffers(row_count, block)
        int_nullmask, int_rows = int_col
        float_nullmask, float_rows = float_col
        bool_nullmask, bool_rows = bool_col

        int_array = pyarrow.Array.from_buffers(pyarrow.int64(), row_count, [
            pyarrow.py_buffer(int_nullmask),
            pyarrow.py_buffer(int_rows),
        ])
        float_array = pyarrow.Array.from_buffers(pyarrow.float64(), row_count, [
            pyarrow.py_buffer(float_nullmask),
            pyarrow.py_buffer(float_rows),
        ])
        bool_array = pyarrow.Array.from_buffers(pyarrow.bool_(), row_count, [
            pyarrow.py_buffer(bool_nullmask),
            pyarrow.py_buffer(bool_rows),
        ])
        return pyarrow.Table.from_arrays([int_array, float_array, bool_array], names=["int_col", "float_col", "bool_col"])
    return message_to_table



def easy_scalars_to_arrow(message):
    row_count = message.avro_rows.row_count
    block = message.avro_rows.serialized_binary_rows
    int_col, float_col, bool_col = _avro_df(row_count, block)
    int_nullmask, int_rows = int_col
    float_nullmask, float_rows = float_col
    bool_nullmask, bool_rows = bool_col

    int_array = pyarrow.Array.from_buffers(pyarrow.int64(), row_count, [
        pyarrow.py_buffer(int_nullmask),
        pyarrow.py_buffer(int_rows),
    ])
    float_array = pyarrow.Array.from_buffers(pyarrow.float64(), row_count, [
        pyarrow.py_buffer(float_nullmask),
        pyarrow.py_buffer(float_rows),
    ])
    bool_array = pyarrow.Array.from_buffers(pyarrow.bool_(), row_count, [
        pyarrow.py_buffer(bool_nullmask),
        pyarrow.py_buffer(bool_rows),
    ])
    return pyarrow.Table.from_arrays([int_array, float_array, bool_array], names=["int_col", "float_col", "bool_col"])


def usa_names_to_arrow(message):
    row_count = message.avro_rows.row_count
    block = message.avro_rows.serialized_binary_rows
    state, gender, year, name, number = _avro_df(row_count, block)
    state_nullmask, state_offsets, state_bytes = state
    year_nullmask, year_rows = year
    number_nullmask, number_rows = number
    # Strings aren't supported. :-( https://issues.apache.org/jira/browse/ARROW-2607
    #my_string_array = pyarrow.Array.from_buffers(pyarrow.string(), row_count, [
    #    pyarrow.py_buffer(my_string_nullmask),
    #    pyarrow.py_buffer(my_string_offsets),
    #    pyarrow.py_buffer(my_string_bytes),
    #])
    year_array = pyarrow.Array.from_buffers(pyarrow.int64(), row_count, [
        pyarrow.py_buffer(year_nullmask),
        pyarrow.py_buffer(year_rows),
    ])
    number_array = pyarrow.Array.from_buffers(pyarrow.int64(), row_count, [
        pyarrow.py_buffer(number_nullmask),
        pyarrow.py_buffer(number_rows),
    ])
    return pyarrow.Table.from_arrays([year_array, number_array], names=["year", "number"])


@numba.jit(nopython=True, nogil=True)
def _copy_bytes(input_bytes, input_start, output_bytes, output_start, strlen):
    input_pos = input_start
    output_pos = output_start
    input_end = input_start + strlen
    output_end = input_start + strlen
    while input_pos < input_end:
        output_bytes[output_pos] = input_bytes[input_pos]
        input_pos += 1
        output_pos += 1


@numba.jit(nopython=True, nogil=True)
def _read_boolean(position, block):
    """Read a single byte whose value is either 0 (false) or 1 (true).

    Returns:
        Tuple[int, numba.uint8]:
            (new position, boolean)

    """
    # We store bool as a bit array. Return 0xff so that we can bitwise AND with
    # the mask that says which bit to write to.
    value = numba.uint8(0xff if block[position] != 0 else 0)
    return (position + 1, value)


@numba.jit(nopython=True, nogil=True)
def _read_bytes(position, block):
    position, strlen = _read_long(position, block)
    value = numpy.empty(strlen, dtype=numpy.uint8)
    for i in range(strlen):
        value[i] = block[position + i]
    return (position + strlen, value)


@numba.jit(nopython=True, nogil=True)
def _read_double(position, block):
    """A double is written as 8 bytes.

    Returns:
        Tuple[numba.int, numba.float64]:
            (new position, double precision floating point)
    """
    # Temporarily use an integer data type for bit shifting purposes. Encoded
    # as little-endian IEEE 754 floating point.
    value = numpy.uint64(block[position])
    value = (value
            | (numpy.uint64(block[position + 1]) << 8)
            | (numpy.uint64(block[position + 2]) << 16)
            | (numpy.uint64(block[position + 3]) << 24)
            | (numpy.uint64(block[position + 4]) << 32)
            | (numpy.uint64(block[position + 5]) << 40)
            | (numpy.uint64(block[position + 6]) << 48)
            | (numpy.uint64(block[position + 7]) << 56))
    return (position + 8, numpy.uint64(value).view(numpy.float64))


@numba.jit(nopython=True, nogil=True)
def _read_long(position, block):
    """Read an int64 using variable-length, zig-zag coding.

    Returns:
        Tuple[int, int]:
            (new position, long integer)
    """
    b = block[position]
    n = b & 0x7F
    shift = 7

    while (b & 0x80) != 0:
        position += 1
        b = block[position]
        n |= (b & 0x7F) << shift
        shift += 7

    return (position + 1, (n >> 1) ^ -(n & 1))


@numba.jit(nopython=True, nogil=True)
def _make_nullmask(row_count):  #, avro_schema):
    extra_byte = 0
    if (row_count % 8) != 0:
        extra_byte = 1
    return numpy.zeros(row_count // 8 + extra_byte, dtype=numpy.uint8)


@numba.jit(nopython=True)  #, nogil=True)
def _rotate_nullmask(nullmask):
    # TODO: Arrow assumes little endian. Detect big endian machines and rotate
    # right, instead.
    nullmask = (nullmask << 1) & 255

    # Have we looped?
    if nullmask == 0:
        # TODO: Detect big endian machines and start at 128, instead.
        return numba.uint8(1)

    return nullmask


#@numba.jit(nopython=True)
#@numba.jit(nopython=True, nogil=True)
def _avro_df(row_count, block):  #, avro_schema):
    """Parse all rows in a stream block.

    Args:
        block ( \
            ~google.cloud.bigquery_storage_v1beta1.types.ReadRowsResponse \
        ):
            A block containing Avro bytes to parse into rows.
        avro_schema (fastavro.schema):
            A parsed Avro schema, used to deserialized the bytes in the
            block.

    Returns:
        Iterable[Mapping]:
            A sequence of rows, represented as dictionaries.
    """
    position = 0
    nullmask = numba.uint8(0)

    int_nullmask = _make_nullmask(row_count)
    float_nullmask = _make_nullmask(row_count)
    bool_nullmask = _make_nullmask(row_count)

    int_col = numpy.empty(row_count, dtype=numpy.int64)
    float_col = numpy.empty(row_count, dtype=numpy.float64)
    bool_col = _make_nullmask(row_count)

    for i in range(row_count):
        nullmask = _rotate_nullmask(nullmask)
        nullbyte = i // 8

        # {
        #     "name": "int_col",
        #     "type": [
        #         "null",
        #         "long"
        #     ]
        # },
        position, union_type = _read_long(position, block)
        if union_type != 0:
            int_nullmask[nullbyte] = int_nullmask[nullbyte] | nullmask
            position, int_col[i] = _read_long(position, block)

        # {
        #     "name": "float_col",
        #     "type": [
        #         "null",
        #         "double"
        #     ]
        # },
        position, union_type = _read_long(position, block)
        if union_type != 0:
            float_nullmask[nullbyte] = float_nullmask[nullbyte] | nullmask
            position, float_col[i] = _read_double(position, block)

        # {
        #     "name": "bool_col",
        #     "type": [
        #         "null",
        #         "boolean"
        #     ]
        # }
        position, union_type = _read_long(position, block)
        if union_type != 0:
            bool_nullmask[nullbyte] = bool_nullmask[nullbyte] | nullmask
            position, boolmask = _read_boolean(position, block)
            bool_col[nullbyte] = bool_col[nullbyte] | (boolmask & nullmask)

    return (
        (int_nullmask, int_col),
        (float_nullmask, float_col),
        (bool_nullmask, bool_col),
    )

