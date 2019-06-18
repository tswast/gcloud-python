
# Copyright 2018 Google LLC
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

"""Just parse, no download."""

from __future__ import absolute_import

import concurrent.futures
import json
import os
import struct
import sys

from google.cloud import bigquery_storage_v1beta1
import avro_to_arrow.generator
import google.cloud.bigquery_storage_v1beta1.types


table_id = "easy_scalars"

with open("benchmark/messages/{}.schema".format(table_id), "r") as schema_file:
    avro_schema = json.load(schema_file)

parser = avro_to_arrow.generator.generate_avro_to_arrow_parser(avro_schema)

tables = []
with open("benchmark/messages/{}.records".format(table_id), "rb") as downloads:
    while True:
        message_len_bytes = downloads.read(4)
        if not message_len_bytes:
            break
        message_len = struct.unpack("i", message_len_bytes)[0]
        message = bigquery_storage_v1beta1.types.ReadRowsResponse()
        message.ParseFromString(downloads.read(message_len))
        tables.append(parser(message))

print(tables[0].to_pandas())
