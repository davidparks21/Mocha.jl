############################################################
# JL Embedding layer (Johnson–Lindenstrauss)
# This layer uses a JL embedding to project the inputs
# down to a lower dimensional space, typically with minimal
# loss of accuracy.
############################################################
@defstruct JLEmbeddingLayer Layer (
  name :: AbstractString = "jlembedding",
  (tops :: Vector{Symbol} = Symbol[], length(tops) > 0),
  (bottoms :: Vector{Symbol} = Symbol[], length(bottoms) == length(tops)),
)

@characterize_layer(JLEmbeddingLayer,
  can_do_bp => true
)

type JLEmbeddingLayerState <: LayerState
  layer      :: IdentityLayer
  blobs      :: Vector{Blob}
  blobs_diff :: Vector{Blob}
end

function setup(backend::Backend, layer::IdentityLayer, inputs::Vector{Blob}, diffs::Vector{Blob})
  blobs      = inputs[:] # shallow copy
  blobs_diff = diffs[:] # shallow_copy

  IdentityLayerState(layer, blobs, blobs_diff)
end

function shutdown(backend::Backend, state::IdentityLayerState)
end

function forward(backend::Backend, state::IdentityLayerState, inputs::Vector{Blob})
  # do nothing
end

function backward(backend::Backend, state::IdentityLayerState, inputs::Vector{Blob}, diffs::Vector{Blob})
  # do nothing
end
