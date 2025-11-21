#!/usr/bin/env python3
"""
Script to structure a transcription using Claude Sonnet API
Reads a raw transcription and creates a well-organized markdown document
"""

import anthropic
import os
import sys
from pathlib import Path
from datetime import datetime


def read_transcription(file_path):
    """Read the transcription file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)


def structure_with_claude(transcription_text, api_key):
    """Use Claude Sonnet to structure the transcription"""
    
    client = anthropic.Anthropic(api_key=api_key)
    
    prompt = f"""Voici une transcription d'une conversation en français. Ta tâche est de créer un document markdown bien structuré à partir de cette transcription.

Instructions:
1. Identifie les principaux sujets abordés dans la conversation
2. Crée une structure claire avec des titres (##) et sous-titres (###)
3. Réorganise le contenu de manière logique sous chaque section
4. Utilise AUTANT QUE POSSIBLE le contenu original de la transcription
5. Conserve les détails importants, les noms de personnes, les organisations mentionnées
6. Améliore la lisibilité en reformulant légèrement si nécessaire, mais garde le sens exact
7. Ajoute une introduction et une conclusion si pertinent
8. Utilise des listes à puces (-) quand approprié pour clarifier les points
9. Mets en gras (**texte**) les termes importants ou noms propres

Format du markdown:
- Titre principal: # Titre
- Sections: ## Titre de section
- Sous-sections: ### Titre de sous-section
- Points importants en gras: **texte**
- Listes: • élément

Transcription à structurer:

{transcription_text}

Crée maintenant le document markdown structuré en français."""

    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=8000,
            temperature=0.3,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        return message.content[0].text
    
    except anthropic.APIError as e:
        print(f"API Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error calling Claude API: {e}")
        sys.exit(1)


def save_markdown(content, output_path):
    """Save the structured content to a markdown file"""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"\n✓ Markdown file created successfully: {output_path}")
    except Exception as e:
        print(f"Error saving file: {e}")
        sys.exit(1)


def main():
    """Main function"""
    
    # Check for API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set.")
        print("Set it with: export ANTHROPIC_API_KEY='your-api-key'")
        sys.exit(1)
    
    # Get input file path
    if len(sys.argv) < 2:
        print("Usage: python structure_transcription.py <transcription_file>")
        print("Example: python structure_transcription.py transcription.txt")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    # Generate output filename
    input_path = Path(input_file)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = input_path.parent / f"{input_path.stem}_structured_{timestamp}.md"
    
    print(f"Reading transcription from: {input_file}")
    transcription = read_transcription(input_file)
    
    print(f"Transcription length: {len(transcription)} characters")
    print("\nStructuring with Claude Sonnet...")
    
    structured_content = structure_with_claude(transcription, api_key)
    
    print(f"\nStructured content length: {len(structured_content)} characters")
    
    save_markdown(structured_content, output_file)
    
    print(f"\nDone! You can now view the structured document.")


if __name__ == "__main__":
    main()