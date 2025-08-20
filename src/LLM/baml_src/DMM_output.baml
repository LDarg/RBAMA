class DMM_Output {
    position int[]
    reasoning string
}

class Position {
    row int
    column int
}

class Positions {
    positions Position[]
}

class State {
    agent_position Position
    person_positions Positions
}

function TestWaterTiles(agent_position: Position, water_tiles: Positions, goal_position: Position) -> Position {
    client CustomGPT3Turbo
    prompt #"return a random row and column
         {{ ctx.output_format }} 
    "#
}

//function ChooseAction(state: State, goal_position: Position, water_tiles: Positions) -> DMM_Output {
function ChooseActionV1(agent_position: Position, goal_position: Position, water_tiles: Positions, grid_size: int, hint: string) -> DMM_Output {
  client CustomGPT3Turbo 
  prompt #"
    You are navigating a gridworld represented in cartesian coordinares. Coordinates start at 0, the lower-left-corner is [0,0].
    The grid is quadratic and of size {{ grid_size }}
    
    Your current position is:
    Column: {{ agent_position.column }} 
    Row: {{ agent_position.row }} 

    Your goal is to reach:
    Column: {{ goal_position.column }} 
    Row: {{ goal_position.row }}

    You can take the following actions:
        "right"
        "down"
        "left"
        "up"
    "right" means that you end up on the coordinates [column+1, row] 
    "down" means that you end up on the coordinates [column, row-1] 
    "left" means that you end up on the coordinates [column-1, row] 
    "up" means that you end up on the coordinates [column, row+1] 

    The following are coordinates for water, which you must avoid entering:
        {{ water_tiles.positions }}

    RULES:
    1. Never choose an action that would move into a water tile.
    2. Output the integer of your chosen action.
    3. Output the reasoning behind your choice of action.

    Make a complete plan containing a complete sequence of single steps of a path that leads to to the goal and then execute the plan. 
    For each step, reason exactly about the consequences of the next action you take: where will you end up and can you move there? make a new plan if you answer the last question with no.
    DO NOT make "jumps", meaning: the sequence must be executable as sequence of primitve actions
    Here are things, you should be especially aware of:
    {{ hint }}

    Specify the next position you want to move to:

    {{ ctx.output_format }} 
  "#
}

function ChooseActionV2(state: State, goal_position: Position, water_tiles: Positions, grid_size: int, hint: string) -> DMM_Output {
  client CustomGPT3Turbo 
  prompt #"
    You are navigating a gridworld represented in cartesian coordinares. Coordinates start at 0, the lower-left-corner is [0,0].
    The grid is quadratic and of size {{ grid_size }}
    
    Your current position is:
    Column: {{ state.agent_position.column }} 
    Row: {{ state.agent_position.row }} 

    Your goal is to reach:
    Column: {{ goal_position.column }} 
    Row: {{ goal_position.row }}

    You can take the following actions:
        "right"
        "down"
        "left"
        "up"
    "right" means that you end up on the coordinates [column+1, row] 
    "down" means that you end up on the coordinates [column, row-1] 
    "left" means that you end up on the coordinates [column-1, row] 
    "up" means that you end up on the coordinates [column, row+1] 

    The following are coordinates for water, which you must avoid entering:
        {{ water_tiles.positions }}

    RULES:
    1. Never choose an action that would move into a water tile.
    2. Output the integer of your chosen action.
    3. Output the reasoning behind your choice of action.

    Make a complete plan containing a complete sequence of single steps of a path that leads to to the goal and then execute the plan. 
    For each step, reason exactly about the consequences of the next action you take: where will you end up and can you move there? make a new plan if you answer the last question with no.
    DO NOT make "jumps", meaning: the sequence must be executable as sequence of primitve actions
    Here are things, you should be especially aware of:
    {{ hint }}

    Specify the next position you want to move to:

    {{ ctx.output_format }} 
  "#
}

function ExtractAction(coordinates: int[], goal_position: int[], water_tiles: int[]) -> DMM_Output {
  // Specify a client as provider/model-name
  // you can use custom LLM params with a custom client name from clients.baml like "client CustomHaiku"
  client CustomGPT3Turbo // Set OPENAI_API_KEY to use this client.
  prompt #"
    You navigate through a gridworld and you are currently on the coordinates 
    {{ coordinates }}
    and you want to navigate to 
    {{ goal_position }}
    the action space (actions you can take) is the following:
        0:  "right",
        1:  "down",
        2:  "left",
        3:  "up",
        4:  "pull out of water" 
    The following are water tiles you must not enter: 
    {{ water_tiles }}

    {{ ctx.output_format }} 
  "#
}

function ExtractActionGoalInformation(coordinates: int[], goal_position: int[], water_tiles: int[]) -> DMM_Output {
  // Specify a client as provider/model-name
  // you can use custom LLM params with a custom client name from clients.baml like "client CustomHaiku"
  client CustomGPT3Turbo // Set OPENAI_API_KEY to use this client.
  prompt #"
    You navigate through a gridworld and you are currently on the coordinates 
    {{ coordinates }}
    and you want to navigate to 
    {{ goal_position }}
    the action space (actions you can take) is the following:
        0:  "right",
        1:  "down",
        2:  "left",
        3:  "up",
        4:  "pull out of water" 
    The following are water tiles you must not enter: 
    {{ water_tiles }}

    {{ ctx.output_format }} 
  "#
}

class Test_Coordinates {
  x int
  y int
}

function ExtractCoordinates(coordinates: int[]) -> Test_Coordinates {
  // Specify a client as provider/model-name
  // you can use custom LLM params with a custom client name from clients.baml like "client CustomHaiku"
  client CustomGPT3Turbo // Set OPENAI_API_KEY to use this client.
  prompt #"
    Print the x and the y coordinate of these:
    {{ coordinates }}
    Action 1 means moving one step downward. 
    {{ ctx.output_format }} 
  "#
}

test coordinate_test {
  functions [ExtractCoordinates]
  args {
    coordinates [1,1]
  }
}