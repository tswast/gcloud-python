
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

from __future__ import absolute_import

import concurrent.futures
import struct
import os

from google.cloud import bigquery_storage_v1beta1


client = bigquery_storage_v1beta1.BigQueryStorageClient()
project_id = os.environ["PROJECT_ID"]
table_ref = bigquery_storage_v1beta1.types.TableReference()

table_ref.project_id = 'swast-scratch'
table_ref.dataset_id = 'schema_examples'
table_ref.table_id = 'easy_scalars'

# table_ref.project_id = 'bigquery-public-data'
# table_ref.dataset_id = 'usa_names'
# table_ref.table_id = 'usa_1910_2013'
# table_ref.dataset_id = 'new_york_citibike'
# table_ref.table_id = 'citibike_trips'

session = client.create_read_session(
    table_ref,
    'projects/{}'.format(project_id),
    requested_streams=1,
)
stream = session.streams[0]
position = bigquery_storage_v1beta1.types.StreamPosition(
    stream=stream,
)
rowstream = client.read_rows(position)

with open("benchmark/messages/{}.records".format(table_ref.table_id), "wb") as downloads:
    for message in rowstream:
        message_bytes = message.SerializeToString()
        message_len = struct.pack("i", len(message_bytes))
        downloads.write(message_len)
        downloads.write(message_bytes)
