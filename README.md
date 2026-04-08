![cdlogo](https://carefuldata.com/images/cdlogo.png)

# llm_hunter

This library is a mostly self contained crate for researching large language model files. The only external dependency is `chrono`. The functionality is also generically useful for entropy (randomness of bytes) analysis of large files for tasks such as cryptanalysis and reverse engineering because rather than just providing an overall entropy score, llm_hunter looks for changes in entropy while reading chunks of the file, calling out byte positions where data becomes less random or more random.

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
The analysis is based on conditional matching of byte sequences commonly found in GGUF model files and scientific data formats (HDF5) files.
There is detection for ZIP segments, including attempts at matching Python Pickle and PyTorch segments.

Up to 1 million patterns per item can be reported by default. Giant sized unknown files are safely chunked and streamed so your computer doesn't run out of RAM
and the whole file is read through via "deep" mode `analyze_file_json_deep` and `analyze_file_json_deep_pretty`. The regular mode also prevents running out of RAM
and runs faster because it only reads part of the start of the file, unless the file is smaller than the sample size, which would result in the whole file being
read in default quick mode.

Here is an example where the target file is _not_ an LLM file, but a linux ELF file for the program `cmake`. Notice the false positives
for "model_family_token". This is a common category of false positive because some of the patterns are small so they com up more frequently
in "random" data. If you are unsure, view the data around that byte position (offset) and find out.

```
{
  "ok": true,
  "file_name": "cmake",
  "created_at_utc": "2026-04-08T00:28:05Z",
  "byte_count": 9245840,
  "scanned_byte_count": 9245840,
  "is_probably_text": false,
  "entropy_sample": 2.215927,
  "signatures": [],
  "detected_specs": [],
  "detected_models": [],
  "matched_patterns": [
    {
      "category": "model_family_token",
      "source": "binary string scan",
      "pattern_matches": [
        {
          "pattern": "t5",
          "offset": 7589202
        },
        {
          "pattern": "t5",
          "offset": 7654995
        },
        {
          "pattern": "t5",
          "offset": 8151524
        },
        {
          "pattern": "t5",
          "offset": 8192136
        },
        {
          "pattern": "t5",
          "offset": 8193812
        },
        {
          "pattern": "t5",
          "offset": 8196376
        },
        {
          "pattern": "t5",
          "offset": 8197280
        },
        {
          "pattern": "t5",
          "offset": 8200716
        },
        {
          "pattern": "t5",
          "offset": 8206200
        },
        {
          "pattern": "t5",
          "offset": 8208904
        },
        {
          "pattern": "t5",
          "offset": 8220704
        },
        {
          "pattern": "t5",
          "offset": 8221656
        },
        {
          "pattern": "t5",
          "offset": 8327472
        }
      ]
    }
  ],
  "detected_data_structures": [],
  "quantization": [],
  "dataset_size": [],
  "parameter_data": [],
  "shapes": [
    {
      "kind": "entropy_transition",
      "offset": 16384,
      "description": "entropy changed by 3.229 between adjacent windows",
      "source": "deep entropy scan",
      "details": {
        "current_entropy": "5.002973",
        "delta": "3.228831",
        "previous_entropy": "1.774142"
      }
    },
    {
      "kind": "entropy_transition",
      "offset": 180224,
      "description": "entropy changed by 2.436 between adjacent windows",
      "source": "deep entropy scan",
      "details": {
        "current_entropy": "4.529191",
        "delta": "2.436045",
        "previous_entropy": "2.093146"
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
{"ok":true,"file_name":"mistral-7b-instruct-v0.1.Q4_0.gguf","created_at_utc":"2026-04-08T00:22:32Z","byte_count":4108916384,"scanned_byte_count":524288,"is_probably_text":false,"entropy_sample":3.367063,"signatures":["magic:GGUF"],"detected_specs":[{"name":"GGUF","version":"2","source":"binary header","notes":["tensor_count=291","kv_count=20"],"pattern_matches":[{"pattern":"GGUF","offset":0}]}],"detected_models":[{"family":"LLaMA","variant":null,"source":"GGUF text region","pattern_matches":[{"pattern":"general.architecture","offset":32},{"pattern":"architecture","offset":40},{"pattern":"llama","offset":64},{"pattern":"general.name","offset":77},{"pattern":"context_length","offset":149}]},{"family":"Mistral","variant":"mistral 7b","source":"GGUF text region","pattern_matches":[{"pattern":"general.name","offset":77},{"pattern":"mistral","offset":111},{"pattern":"context_length","offset":149},{"pattern":"embedding_length","offset":185}]}],"matched_patterns":[{"category":"model_family_token","source":"GGUF text region","pattern_matches":[{"pattern":"phi","offset":41833},{"pattern":"bert","offset":61157},{"pattern":"phi","offset":112308},{"pattern":"bert","offset":294134},{"pattern":"bloom","offset":344071},{"pattern":"t5","offset":508797},{"pattern":"t5","offset":508829}]}],"detected_data_structures":[{"name":"gguf_header","source":"binary header","offset":0,"length":24,"details":{"kv_count":"20","tensor_count":"291"},"pattern_matches":[{"pattern":"GGUF","offset":0}]},{"name":"gguf_kv_region","source":"GGUF key/value strings","offset":32,"length":null,"details":{},"pattern_matches":[{"pattern":"general.architecture","offset":32},{"pattern":"llama.context_length","offset":143},{"pattern":"llama.embedding_length","offset":179}]}],"quantization":[{"scheme":"ggml","source":"GGUF text region","pattern_matches":[{"pattern":"ggml","offset":553},{"pattern":"ggml","offset":598},{"pattern":"ggml","offset":461313}]},{"scheme":"gguf","source":"GGUF text region","pattern_matches":[{"pattern":"gguf","offset":0}]}],"dataset_size":[],"parameter_data":[],"shapes":[],"metadata":{"configured_max_safetensors_header_bytes":"16777216","configured_scan_window_bytes":"524288","gguf_kv_count":"20","gguf_tensor_count":"291","partial_scan":"true","pretty_json":"false","scan_strategy":"prefix_window"},"warnings":["safetensors header length 9769928519 exceeds configured max 16777216; skipping detailed parse"]}
```

## Using llm_hunter

The source code and license can be copied and included in another project or llm_hunter can be installed from crates.io to an existing cargo project:

```
cargo add llm_hunter
```

This library is maintained as best as is reasonable. 

Testing and adoption are currently in progress, as of April 6th 2026.

Also see [giant-spellbook](https://github.com/jpegleg/giant-spellbook/), a CLI tool for cryptanalysis and binary analysis.
