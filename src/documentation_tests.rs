use std::{fs, path::Path};

#[test]
fn tensor_parameters_have_shape_documentation() {
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut rust_files = Vec::new();
    collect_rust_files(workspace, &mut rust_files);
    let mut missing = Vec::new();

    for path in rust_files {
        let source = fs::read_to_string(&path).unwrap();
        let lines = source.lines().collect::<Vec<_>>();
        for (line_index, line) in lines.iter().enumerate() {
            let trimmed = line.trim_start();
            if trimmed.starts_with("//") || trimmed.starts_with('*') {
                continue;
            }
            let Some(function_offset) = line.find("fn ") else {
                continue;
            };
            let signature = lines[line_index..lines.len().min(line_index + 35)].join("\n");
            let Some(parameters) = function_parameters(&signature, function_offset) else {
                continue;
            };
            if !parameters.contains("Tensor") {
                continue;
            }

            let documentation = preceding_documentation(&lines, line_index);
            if documentation.is_empty()
                || !(documentation.contains("shape") || documentation.contains('['))
            {
                missing.push(format!(
                    "{}:{}",
                    path.strip_prefix(workspace).unwrap().display(),
                    line_index + 1
                ));
            }
        }
    }

    assert!(
        missing.is_empty(),
        "tensor parameters require a doc comment describing their shape:\n{}",
        missing.join("\n")
    );
}

fn collect_rust_files(directory: &Path, output: &mut Vec<std::path::PathBuf>) {
    for entry in fs::read_dir(directory).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.is_dir() {
            let name = path.file_name().and_then(|name| name.to_str());
            if !matches!(name, Some("target" | ".git")) {
                collect_rust_files(&path, output);
            }
        } else if path.extension().and_then(|extension| extension.to_str()) == Some("rs") {
            output.push(path);
        }
    }
}

fn function_parameters(signature: &str, function_offset: usize) -> Option<&str> {
    let opening = signature[function_offset..].find('(')? + function_offset;
    let mut depth = 0usize;
    for (offset, character) in signature[opening..].char_indices() {
        match character {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    return Some(&signature[opening + 1..opening + offset]);
                }
            }
            _ => {}
        }
    }
    None
}

fn preceding_documentation(lines: &[&str], line_index: usize) -> String {
    let mut index = line_index;
    while index > 0 {
        let previous = lines[index - 1].trim_start();
        if previous.is_empty() || previous.starts_with("#[") {
            index -= 1;
        } else {
            break;
        }
    }

    let mut documentation = Vec::new();
    while index > 0 {
        let previous = lines[index - 1].trim_start();
        let Some(comment) = previous.strip_prefix("///") else {
            break;
        };
        documentation.push(comment);
        index -= 1;
    }
    documentation.reverse();
    documentation.join(" ").to_lowercase()
}
