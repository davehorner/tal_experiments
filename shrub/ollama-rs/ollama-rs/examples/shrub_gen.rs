use std::path::PathBuf;

use ollama_rs::{coordinator::Coordinator, generation::chat::ChatMessage, Ollama};

/// Get the CPU temperature in Celsius.
#[ollama_rs::function]
async fn get_unxtal_quoted_string(string: String) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    let quoted = string.replace(' ', " 20 ");
    Ok(format!("\"{}\"", quoted))
}

/// Get the available space in bytes for a given path.
///
/// * path - Path to check available space for.
#[ollama_rs::function]
async fn get_available_space(
    path: PathBuf,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    Ok(fs2::available_space(path).map_or_else(
        // Note: this will let LLM handle the error. Return `Err` if you want to bubble it up.
        |err| format!("failed to get available space: {err}"),
        |space| space.to_string(),
    ))
}

/// Get the weather for a given city.
///
/// * city - City to get the weather for.
#[ollama_rs::function]
async fn get_weather(city: String) -> Result<String, Box<dyn std::error::Error + Sync + Send>> {
    Ok(reqwest::get(format!("https://wttr.in/{city}?format=%C+%t"))
        .await?
        .text()
        .await?)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Sync + Send>> {
    let ollama = Ollama::default();
    let history = vec![];
    let mut coordinator = Coordinator::new(ollama, "llama3.2".to_string(), history)
        .add_tool(get_unxtal_quoted_string)
        .add_tool(get_available_space)
        .add_tool(get_weather);

    let user_messages = vec![
        r#"
        create me a uxn varvara tal source that has the ascii 
        art of a shrub embedded in it. it should print the shrub to the console when run.
        here is the example hello world tal file for reference:
        ```uxntal
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
        ```
        ----
        # TAL/UXN assembly  syntax for strings.

## Delimiters

- open: `"`
- close: any byte <= 0x20 ( ASCII space)

The content between the delimiters is emitted as raw ASCII bytes.

To emit a double quote character, use two consecutive quotes followed
by a space: `"" ` (emits a single `"` byte).
Strings can be placed anywhere tokens are allowed (e.g., after
instructions, as data).

## Examples:

- `"HELLO ` → emits bytes for `HELLO`.
- `""HELP" ` → emits bytes for `"HELP"`.
- `"" ` → emits a single `"` byte.

No label or macro expansion occurs inside string literals.
        only respond with the source code, nothing else.
        you should modify the source code to include the ascii art of a shrub using uxntal strings and the tool get_unxtal_quoted_string"#,
    ];

    for user_message in user_messages {
        println!("User: {user_message}");

        let user_message = ChatMessage::user(user_message.to_owned());
        let resp = coordinator.chat(vec![user_message]).await?;
        println!("Assistant: {}", resp.message.content);
    }

    Ok(())
}
