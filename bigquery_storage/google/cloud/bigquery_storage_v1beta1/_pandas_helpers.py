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

"""Helpers to parse blocks into pandas DataFrames."""

# TODO: optional imports
import numpy
import numba
import pandas
import pyarrow


def usa_names_to_dataframe(message, dtypes=None):
    row_count = message.avro_rows.row_count
    block = message.avro_rows.serialized_binary_rows
    # state, gender, year, name, number
    year, number = _avro_df(row_count, block)
    year_nullmask, year_rows = year
    number_nullmask, number_rows = number
    year_array = pyarrow.Array.from_buffers(pyarrow.int64(), row_count, [
        pyarrow.py_buffer(year_nullmask),
        pyarrow.py_buffer(year_rows),
    ])
    number_array = pyarrow.Array.from_buffers(pyarrow.int64(), row_count, [
        pyarrow.py_buffer(number_nullmask),
        pyarrow.py_buffer(number_rows),
    ])
    return pyarrow.Table.from_arrays([year_array, number_array], names=["year", "number"]).to_pandas()


@numba.jit(nopython=True)
def _read_bytes(position, block):
    position, strlen = _read_long(position, block)
    value = numpy.empty(strlen, dtype=numpy.uint8)
    for i in range(strlen):
        value[i] = block[position + i]
    return (position + strlen, value)


@numba.jit(nopython=True)
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


@numba.jit(nopython=True) #, nogil=True)
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


@numba.jit(nopython=True)  #, nogil=True)
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
    #state_nullmask
    #gender_nullmask
    year_nullmask = _make_nullmask(row_count)
    #name_nullmask
    number_nullmask = _make_nullmask(row_count)
    #state
    #gender
    year = numpy.empty(row_count, dtype=numpy.int64)
    #name
    number = numpy.empty(row_count, dtype=numpy.int64)
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
        else:
            position, state = _read_bytes(position, block)
            #position, strlen = _read_long(position, block)
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
        else:
            position, gender = _read_bytes(position, block)
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
        else:
            position, name = _read_bytes(position, block)

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

        # rows.append((state, gender, year, name, number))
    return ((year_nullmask, year), (number_nullmask, number))

