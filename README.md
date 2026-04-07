![cdlogo](https://carefuldata.com/images/cdlogo.png)

# llm_hunter

This library is for researching large language model files. It is also generically useful for entropy (randomness of bytes) analysis of large files for tasks such as cryptanalysis and reverse engineering.

Compact quick JSON analysis, don't parse the whole file:

```
let json = llm_hunter::analyze_file_json("mistral-7b-instruct-v0.1.Q4_0.gguf");
```

Use pretty print JSON output instead of compact JSON:
```
let json = llm_hunter::analyze_file_json_pretty("mistral-7b-instruct-v0.1.Q4_0.gguf");
```

Use "deep" mode to read the entire file:

```
let json = llm_hunter::analyze_file_json_deep_pretty("mistral-7b-instruct-v0.1.Q4_0.gguf");
```

The JSON output includes pattern matching byte offsets, and notes about offsets for a number of potentially interesting aspects.
The analysis is based on conditional matching of byte sequences commonly found in model files.

Here is an example where the target file is _not_ an LLM file, but an unknown ELF file:

```
{
  "ok": true,
  "file_name": "some_file",
  "created_at_utc": "2026-04-07T02:31:05Z",
  "byte_count": 788048,
  "scanned_byte_count": 788048,
  "is_probably_text": false,
  "entropy_sample": 1.602038,
  "signatures": [],
  "detected_specs": [],
  "detected_models": [],
  "matched_patterns": [],
  "detected_data_structures": [],
  "quantization": [],
  "dataset_size": [],
  "parameter_data": [],
  "shapes": [
    {
      "kind": "entropy_transition",
      "offset": 16384,
      "length": 16384,
      "description": "entropy changed by 2.992 between adjacent windows",
      "source": "deep entropy scan",
      "details": {
        "current_entropy": "5.841055",
        "delta": "2.991641",
        "previous_entropy": "2.849414"
      }
    },
    {
      "kind": "entropy_transition",
      "offset": 786432,
      "length": 1616,
      "description": "entropy changed by 3.222 between adjacent windows",
      "source": "deep entropy scan",
      "details": {
        "current_entropy": "1.722589",
        "delta": "3.222235",
        "previous_entropy": "4.944823"
      }
    }
  ],
  "metadata": {
    "configured_deep_entropy_window_bytes": "16384",
    "configured_deep_scan_chunk_bytes": "1048576",
    "configured_deep_scan_overlap_bytes": "16384",
    "pretty_json": "true",
    "scan_strategy": "deep_stream"
  },
  "warnings": []
}
```

<b>Warning</b> large files can take a long time to process. There is some threading to speed up tasks, but large files can still take multiple minutes to process and create large JSON outputs.


Here is an example output section for matching of a gguf file, using the compact JSON mode and quick scan mode:

```
{"ok":true,"file_name":"mistral-7b-instruct-v0.1.Q4_0.gguf","created_at_utc":"2026-04-07T02:46:27Z","byte_count":4108916384,"scanned_byte_count":524288,"is_probably_text":false,"entropy_sample":3.367063,"signatures":["magic:GGUF"],"detected_specs":[{"name":"GGUF","version":"2","source":"binary header","notes":["tensor_count=291","kv_count=20"],"pattern_matches":[{"pattern":"GGUF","offset":0,"length":4}]}],"detected_models":[{"family":"LLaMA","variant":null,"source":"GGUF text region","pattern_matches":[{"pattern":"general.architecture","offset":32,"length":20},{"pattern":"architecture","offset":40,"length":12},{"pattern":"llama","offset":64,"length":5},{"pattern":"general.name","offset":77,"length":12},{"pattern":"context_length","offset":149,"length":14}]},{"family":"Mistral","variant":"mistral 7b","source":"GGUF text region","pattern_matches":[{"pattern":"general.name","offset":77,"length":12},{"pattern":"mistral","offset":111,"length":7},{"pattern":"context_length","offset":149,"length":14},{"pattern":"embedding_length","offset":185,"length":16}]}],"matched_patterns":[{"category":"model_family_token","source":"GGUF text region","pattern_matches":[{"pattern":"phi","offset":41833,"length":3},{"pattern":"bert","offset":61157,"length":4},{"pattern":"phi","offset":112308,"length":3},{"pattern":"bert","offset":294134,"length":4},{"pattern":"bloom","offset":344071,"length":5},{"pattern":"t5","offset":508797,"length":2},{"pattern":"t5","offset":508829,"length":2}]}],"detected_data_structures":[{"name":"gguf_header","source":"binary header","offset":0,"length":24,"details":{"kv_count":"20","tensor_count":"291"},"pattern_matches":[{"pattern":"GGUF","offset":0,"length":4}]},{"name":"gguf_kv_region","source":"GGUF key/value strings","offset":32,"length":null,"details":{},"pattern_matches":[{"pattern":"general.architecture","offset":32,"length":20},{"pattern":"llama.context_length","offset":143,"length":20},{"pattern":"llama.embedding_length","offset":179,"length":22}]}],"quantization":[{"scheme":"ggml","source":"GGUF text region","pattern_matches":[{"pattern":"ggml","offset":553,"length":4},{"pattern":"ggml","offset":598,"length":4},{"pattern":"ggml","offset":461313,"length":4}]},{"scheme":"gguf","source":"GGUF text region","pattern_matches":[{"pattern":"gguf","offset":0,"length":4}]}],"dataset_size":[],"parameter_data":[],"shapes":[],"metadata":{"configured_max_safetensors_header_bytes":"16777216","configured_scan_window_bytes":"524288","gguf_kv_count":"20","gguf_tensor_count":"291","partial_scan":"true","pretty_json":"false","scan_strategy":"prefix_window"},"warnings":["safetensors header length 9769928519 exceeds configured max 16777216; skipping detailed parse"]}
```

## Using llm_hunter

The source code and license can be copied and included in another project or llm_hunter can be installed from crates.io to an existing cargo project:

```
cargo add llm_hunter
```

This library is maintained as best as is reasonable. 

Testing and adoption are currently in progress, as of April 6th 2026.

Also see [giant-spellbook](https://github.com/jpegleg/giant-spellbook/), a CLI tool for cryptanalysis and binary analysis.
