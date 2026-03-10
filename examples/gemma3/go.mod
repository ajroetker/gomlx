module github.com/gomlx/gomlx/examples/gemma3

go 1.26

require (
	github.com/ajroetker/huggingface-gomlx v0.0.0
	github.com/gomlx/go-huggingface v0.3.2-0.20260125064416-b0f56ca7fbef
	github.com/gomlx/gomlx v0.26.1-0.20260310154955-4e81b35dea62
	github.com/gomlx/onnx-gomlx v0.3.5-0.20260222061411-0ce0c531c49d
	k8s.io/klog/v2 v2.140.0
)

require (
	github.com/ajroetker/go-highway v0.0.12 // indirect
	github.com/ajroetker/gollmx v0.0.0-20260310024938-f1f27ec7a1c3 // indirect
	github.com/dustin/go-humanize v1.0.1 // indirect
	github.com/eliben/go-sentencepiece v0.7.0 // indirect
	github.com/go-logr/logr v1.4.3 // indirect
	github.com/gofrs/flock v0.13.0 // indirect
	github.com/gomlx/exceptions v0.0.3 // indirect
	github.com/gomlx/go-xla v0.1.5-0.20260310165402-a903b9c73d3d // indirect
	github.com/google/uuid v1.6.0 // indirect
	github.com/pkg/errors v0.9.1 // indirect
	github.com/x448/float16 v0.8.4 // indirect
	golang.org/x/exp v0.0.0-20260218203240-3dfff04db8fa // indirect
	golang.org/x/sys v0.42.0 // indirect
	golang.org/x/term v0.40.0 // indirect
	golang.org/x/text v0.34.0 // indirect
	google.golang.org/protobuf v1.36.11 // indirect
)

replace (
	github.com/ajroetker/huggingface-gomlx => ../../../huggingface-gomlx
	github.com/gomlx/gomlx => ../..
)
