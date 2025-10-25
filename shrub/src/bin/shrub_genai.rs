

use genai::Client;
use genai::chat::printer::{PrintChatStreamOptions, print_chat_stream};
use genai::chat::{ChatMessage, ChatRequest};
use tracing_subscriber::EnvFilter;
use uxn_tal::Assembler;

const MODEL_OPENAI: &str = "gpt-4o-mini"; // o1-mini, gpt-4o-mini
const MODEL_ANTHROPIC: &str = "claude-3-haiku-20240307";
// or namespaced with simple anme "fireworks::qwen3-30b-a3b", or "fireworks::accounts/fireworks/models/qwen3-30b-a3b"
const MODEL_FIREWORKS: &str = "accounts/fireworks/models/qwen3-30b-a3b";
const MODEL_TOGETHER: &str = "together::openai/gpt-oss-20b";
const MODEL_GEMINI: &str = "gemini-2.0-flash";
const MODEL_GROQ: &str = "llama-3.1-8b-instant";
const MODEL_OLLAMA: &str = "codellama:7b"; // sh: `ollama pull gemma:2b`
const MODEL_XAI: &str = "grok-3-mini";
const MODEL_DEEPSEEK: &str = "deepseek-chat";
const MODEL_ZAI: &str = "glm-4-plus";
const MODEL_COHERE: &str = "command-r7b-12-2024";

// NOTE: These are the default environment keys for each AI Adapter Type.
//       They can be customized; see `examples/c02-auth.rs`
const MODEL_AND_KEY_ENV_NAME_LIST: &[(&str, &str)] = &[
	// -- De/activate models/providers
	(MODEL_OPENAI, "OPENAI_API_KEY"),
	(MODEL_ANTHROPIC, "ANTHROPIC_API_KEY"),
	(MODEL_GEMINI, "GEMINI_API_KEY"),
	(MODEL_FIREWORKS, "FIREWORKS_API_KEY"),
	(MODEL_TOGETHER, "TOGETHER_API_KEY"),
	(MODEL_GROQ, "GROQ_API_KEY"),
	(MODEL_XAI, "XAI_API_KEY"),
	(MODEL_DEEPSEEK, "DEEPSEEK_API_KEY"),
	(MODEL_OLLAMA, ""),
	(MODEL_ZAI, "ZAI_API_KEY"),
	(MODEL_COHERE, "COHERE_API_KEY"),
];

// NOTE: Model to AdapterKind (AI Provider) type mapping rule
//  - starts_with "gpt"      -> OpenAI
//  - starts_with "claude"   -> Anthropic
//  - starts_with "command"  -> Cohere
//  - starts_with "gemini"   -> Gemini
//  - model in Groq models   -> Groq
//  - starts_with "glm"      -> ZAI
//  - For anything else      -> Ollama
//
// This can be customized; see `examples/c03-mapper.rs`
fn strip_uxntal_code_block(s: &str) -> String {
	let s = s.trim();
	let s = s.strip_prefix("```uxntal").or_else(|| s.strip_prefix("```tal")).unwrap_or(s);
	let s = s.strip_suffix("```").unwrap_or(s);
	s.trim().to_string()
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
	tracing_subscriber::fmt()
		.with_env_filter(EnvFilter::new("genai=debug"))
		// .with_max_level(tracing::Level::DEBUG) // To enable all sub-library tracing
		.init();

	let question = r###"
	create me a uxn varvara tal source that has the ascii art of a shrub embedded in it.
	 it should print the shrub to the console when run. 
	 
	 here is the example hello world tal file for reference:
        ( dev/console )
        |10 @Console [ &pad $8 &char ]

        ( init )

        |0100 ( -> )

            ;hello-world

            &loop
                ( send ) LDAk .Console/char DEO
                ( incr ) #0001 ADD2
                ( loop ) LDAk ,&loop JCN
            POP2

        BRK

        @hello-world "Hello 20 "World!
---- 
The program should start with a comment and end with BRK.
# TAL/UXN assembly syntax for strings. 
## String Delimiters for uxntal strings
- a single " open starts a string. "Dave indicates the start of a string Dave.
- close: any byte <= 0x20 ( ASCII space)
String are not terminated by ", they start with " and end with the first byte <= 0x20.
To emit a double quote character, use two consecutive quotes followed by a space: "" (emits a single " byte).
Newlines are encoded as byte 0x0A.
Spaces should be encoded as 0x20.
## Examples:
 - "HELLO → emits bytes for HELLO.
 - ""HELP" → emits bytes for "HELP".
 - "" → emits a single " byte.
 - "HELLO 20 "World!  → emits bytes for HELLO, a space (0x20), and World!
No label or macro expansion occurs inside string literals.
There are no multi-line strings; each line must be separately quoted.
There is no \n escape sequence.  multiple spaces require multiple 0x20 bytes.

only respond with the source code, nothing else, no headers or markdown ticks.
you should modify the source code to include the ascii art of a shrub using uxntal strings
	make sure that the strings you generate follow these rules.
"###;

	let chat_req = ChatRequest::new(vec![
		// -- Messages (de/activate to see the differences)
		ChatMessage::system("Answer in one sentence"),
		ChatMessage::user(question),
	]);

	let client = Client::default();

	let print_options = PrintChatStreamOptions::from_print_events(false);

	for (model, env_name) in MODEL_AND_KEY_ENV_NAME_LIST {
		// Skip if the environment name is not set
		if !env_name.is_empty() && std::env::var(env_name).is_err() {
			println!("===== Skipping model: {model} (env var not set: {env_name})");
			continue;
		}

		let adapter_kind = client.resolve_service_target(model).await?.model.adapter_kind;

		println!("\n===== MODEL: {model} ({adapter_kind}) =====");

		println!("\n--- Question:\n{question}");

		println!("\n--- Answer:");
		let chat_res = client.exec_chat(model, chat_req.clone(), None).await?;
		println!("{}", chat_res.first_text().unwrap_or("NO ANSWER"));

		match chat_res.first_text() {
			Some(answer) => {
				let answer = strip_uxntal_code_block(&answer);
				if answer.contains("```uxntal") {
					let parts: Vec<&str> = answer.split("```uxntal").collect();
					if parts.len() > 1 {
						let code_parts: Vec<&str> = parts[1].split("```").collect();
						if !code_parts.is_empty() {
							let tal_code = code_parts[0];
							println!("\n--- Extracted TAL Code:\n{}", tal_code);

							// Optionally, assemble the TAL code to verify correctness
							let mut assembler = Assembler::new();
							match assembler.assemble(tal_code, None) {
								Ok(_) => println!("TAL code assembled successfully."),
								Err(e) => println!("Error assembling TAL code: {}", e),
							}
						}
					}
				}
				println!("\n--- Extracted TAL Code:\n{}", answer);

							// Optionally, assemble the TAL code to verify correctness
				let mut assembler = Assembler::new();
				match assembler.assemble(&answer, None) {
					Ok(_) => println!("TAL code assembled successfully."),
					Err(e) => println!("Error assembling TAL code: {}", e),
				}
			}
			None => println!("No text answer to extract TAL code from."),
		}

		println!("\n--- Answer: (streaming)");
		let chat_res = client.exec_chat_stream(model, chat_req.clone(), None).await?;
		print_chat_stream(chat_res, Some(&print_options)).await?;

		println!();
	}

	Ok(())
}
