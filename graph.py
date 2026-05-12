import json
import subprocess

# Path to the raw JSON file containing Neo4j export data
raw_json_file = "records.json"  # Replace with the actual file path

# Function to extract schema from the JSON data
def extract_schema(json_file):
    try:
        schema = {"nodes": {}, "relationships": set()}
        
        # Open the JSON file and process it line by line
        with open(json_file, "r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                try:
                    # Strip extra whitespace and parse the JSON line
                    line = line.strip()
                    if not line:  # Skip empty lines
                        continue
                    
                    record = json.loads(line)  # Parse the JSON object on the current line
                    
                    # Extract node details
                    node = record.get("n")
                    if node:
                        labels = node.get("labels", [])
                        properties = list(node.get("properties", {}).keys())
                        for label in labels:
                            if label not in schema["nodes"]:
                                schema["nodes"][label] = set()
                            schema["nodes"][label].update(properties)

                    # Extract relationship details
                    relationship = record.get("r")
                    if relationship:
                        schema["relationships"].add(relationship.get("type", "UNKNOWN"))
                
                except json.JSONDecodeError as e:
                    print(f"Skipping invalid JSON line at line {line_number}: {line}")
                    continue
                except Exception as e:
                    print(f"Error processing line {line_number}: {e}")
                    continue

        # Convert sets to lists for JSON compatibility
        schema["nodes"] = {k: list(v) for k, v in schema["nodes"].items()}
        schema["relationships"] = list(schema["relationships"])
        return schema

    except FileNotFoundError:
        print(f"JSON file '{json_file}' not found.")
        return None
    except Exception as e:
        print(f"Error processing the JSON file: {e}")
        return None

# Function to generate Cypher query using Ollama
def generate_cypher_query(natural_language_query, schema):
    # Construct the prompt with schema details
    prompt = """
    I have the following schema in Neo4j:
    """
    for node, properties in schema["nodes"].items():
        prompt += f"- {node} nodes with properties: {properties}\n"
    prompt += "Relationships: " + ", ".join(schema["relationships"]) + "\n"

    prompt += f"\nPlease translate the following natural language query into a Cypher query:\n{natural_language_query}"

    # Call Ollama CLI to generate the Cypher query
    try:
        result = subprocess.run(
            ["ollama", "generate", "--model", "cypher-3.1"],
            input=prompt.encode("utf-8"),
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            cypher_query = result.stdout.strip()
            return cypher_query
        else:
            print(f"Error generating Cypher query: {result.stderr}")
            return None
    except FileNotFoundError:
        print("Ollama CLI is not installed or not in PATH.")
        return None

# Main function to process user input and return results
def main():
    # Extract schema from the raw JSON file
    schema = extract_schema(raw_json_file)
    if not schema:
        print("Failed to extract schema. Exiting.")
        return

    # Get the user's query in natural language
    natural_language_query = input("Enter your query in natural language: ")

    # Generate the corresponding Cypher query using Ollama
    cypher_query = generate_cypher_query(natural_language_query, schema)
    if cypher_query:
        print(f"Generated Cypher Query: {cypher_query}")
    else:
        print("Failed to generate Cypher query.")

if __name__ == "__main__":
    main()
