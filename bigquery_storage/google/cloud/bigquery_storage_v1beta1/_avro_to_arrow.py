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
# NUMERIC (decimal)
# FLOAT64
# BOOL
# DATE
# DATETIME - need to parse from string
# TIME
# TIMESTAMP



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
def _read_bytes(position, block):
    position, strlen = _read_long(position, block)
    value = numpy.empty(strlen, dtype=numpy.uint8)
    for i in range(strlen):
        value[i] = block[position + i]
    return (position + strlen, value)


@numba.jit(nopython=True, nogil=True)
def _read_long(position, block):
    """int and long values are written using variable-length, zig-zag
    coding.

    Returns (new position, long integer)

    Derived from fastavro's implementation.
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
@numba.jit(nopython=True, nogil=True)
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

    state_nullmask = _make_nullmask(row_count)
    gender_nullmask = _make_nullmask(row_count)
    year_nullmask = _make_nullmask(row_count)
    name_nullmask = _make_nullmask(row_count)
    number_nullmask = _make_nullmask(row_count)

    state_input_offsets = numpy.empty(row_count, dtype=numpy.int64)
    gender_input_offsets = numpy.empty(row_count, dtype=numpy.int64)
    name_input_offsets = numpy.empty(row_count, dtype=numpy.int64)

    state_offsets = numpy.empty(row_count + 1, dtype=numpy.int64)
    gender_offsets = numpy.empty(row_count + 1, dtype=numpy.int64)
    year = numpy.empty(row_count, dtype=numpy.int64)
    name_offsets = numpy.empty(row_count + 1, dtype=numpy.int64)
    number = numpy.empty(row_count, dtype=numpy.int64)

    state_offsets[0] = 0
    gender_offsets[0] = 0
    name_offsets[0] = 0

    for i in range(row_count):
        nullmask = _rotate_nullmask(nullmask)
        nullbyte = i // 8

        # {
        #     "name": "state",
        #     "type": [
        #         "null",
        #         "string"
        #     ],
        #     "doc": "2-digit state code"
        # }
        position, union_type = _read_long(position, block)
        if union_type == 0:
            state = None
            state_offsets[i + 1] = state_offsets[i]
        else:
            #position, state = _read_bytes(position, block)
            position, strlen = _read_long(position, block)

            # Save where we are to copy later.
            state_input_offsets[i] = position
            state_offsets[i + 1] = state_offsets[i] + strlen
            position = position + strlen

            #state = str(block[position:position + strlen], encoding="utf-8")
            #state = block[position:position + strlen].decode("utf-8")
            #block[i] = block[position] #:position + strlen]
            #state = b"" + block[position:position + strlen]
            #state = "WA"
            #position = position + strlen
        # {
        #     "name": "gender",
        #     "type": [
        #         "null",
        #         "string"
        #     ],
        #     "doc": "Sex (M=male or F=female)"
        # }
        position, union_type = _read_long(position, block)
        if union_type == 0:
            gender = None
            gender_offsets[i + 1] = gender_offsets[i]
        else:
            position, strlen = _read_long(position, block)

            # Save where we are to copy later.
            gender_input_offsets[i] = position
            gender_offsets[i + 1] = gender_offsets[i] + strlen
            position = position + strlen

        #gender = block[position:position + strlen]
        # {
        #     "name": "year",
        #     "type": [
        #         "null",
        #         "long"
        #     ],
        #     "doc": "4-digit year of birth"
        # }
        position, union_type = _read_long(position, block)
        if union_type != 0:
            year_nullmask[nullbyte] = year_nullmask[nullbyte] | nullmask
            position, year[i] = _read_long(position, block)
        # {
        #     "name": "name",
        #     "type": [
        #         "null",
        #         "string"
        #     ],
        #     "doc": "Given name of a person at birth"
        # }
        position, union_type = _read_long(position, block)
        if union_type == 0:
            name = None
            name_offsets[i + 1] = name_offsets[i]
        else:
            position, strlen = _read_long(position, block)

            # Save where we are to copy later.
            name_input_offsets[i] = position
            name_offsets[i + 1] = name_offsets[i] + strlen
            position = position + strlen

        #name = block[position:position + strlen]
        # {
        #     "name": "number",
        #     "type": [
        #         "null",
        #         "long"
        #     ],
        #     "doc": "Number
        # }
        position, union_type = _read_long(position, block)
        if union_type != 0:
            number_nullmask[nullbyte] = number_nullmask[nullbyte] | nullmask
            position, number[i] = _read_long(position, block)

    # Second pass: copy all the strings.
    state = numpy.empty(state_offsets[row_count - 1], dtype=numpy.uint8)
    gender = numpy.empty(gender_offsets[row_count - 1], dtype=numpy.uint8)
    name = numpy.empty(name_offsets[row_count - 1], dtype=numpy.uint8)

    for i in numba.prange(row_count):
        input_start = state_input_offsets[i]
        output_start = state_offsets[i]
        strlen = state_offsets[i + 1] - state_offsets[i]
        _copy_bytes(block, input_start, state, output_start, strlen)
        #_copy_bytes(block, gender_input_offets[i], gender, gender_offsets[i], gender_offsets[i + 1] - gender_offsets[i])
        #_copy_bytes(block, name_input_offets[i], name, name_offsets[i], name_offsets[i + 1] - name_offsets[i])

        # rows.append((state, gender, year, name, number))
    return (
        (state_nullmask, state_offsets, state),
        (gender_nullmask, gender_offsets, None),
        (year_nullmask, year),
        (name_nullmask, name_offsets, None),
        (number_nullmask, number),
    )

