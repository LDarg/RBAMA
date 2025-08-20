// Defining a data model.
class Resume {
  name string
  email string
  experience string[]
  skills string[]
}

// Create a function to extract the resume from a string.
function ExtractResume(resume: string) -> Resume {
  // Specify a client as provider/model-name
  // you can use custom LLM params with a custom client name from clients.baml like "client CustomHaiku"
  client CustomGPT3Turbo // Set OPENAI_API_KEY to use this client.
  prompt #"
    Extract from this content:
    {{ resume }}

    {{ ctx.output_format }} 
  "#
}
// {{ ctx.output_format }} generates a prompt for the LLM to give an output in the resume format


// Test the function with a sample resume. Open the VSCode playground to run this.
test vaibhav_resume {
  functions [ExtractResume]
  args {
    resume #"
      Vaibhav Gupta
      vbv@boundaryml.com

      Experience:
      - Founder at BoundaryML
      - CV Engineer at Google
      - CV Engineer at Microsoft

      Skills:
      - Rust
      - C++
    "#
  }
}
