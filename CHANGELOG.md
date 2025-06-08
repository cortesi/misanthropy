8 June 2025 - v0.0.8

- Support extended thinking, including streaming responses
- Support the text editor tool API
- Return `RateLimitExceeded` error when rate limit is exceeded
- Add `ToolResult::is_error`
- Default model is now Claude Sonnet 4
- Support top_k, top_p and metadata on MessagesRequest
- Update dependencies


28 August 2024 - v0.0.7

- Add the `cache_control` field to various API objects. 
- `MessagesResponse.system` is now a vec of `Content` objects.
- Add cache_creation_input_tokens and cache_read_input_tokens to `Usage`
- `MessagesResponse.usage` is now cumulative for streamed responses 
- Fix ToolChoice::SpecifiedTool structure (@philpax)
- Add support for error events in streaming responses

28 July 2024 - v0.0.6

- Support beta header to enable extended token output. See:
    https://docs.anthropic.com/en/release-notes/api#july-15th-2024
- misan chat: Interactive chat command
- misan: --json flag to output responses as JSON
- @philpax: Improved error responses
- MessagesRequest.merge_response and merge_streamed_response to merge responses 
  into requests 
- Add support for specifying stop tokens


01 June 2024 - v0.0.5

- Make `Anthropic.messages` and `Anthropic.messages_stream` take a reference to `MessagesRequest`
- @philpax: Add support for tool responses
