class NavigationAgent {
  tool_name "navigation_agent"
  goal_position_coordinates int[] @description("the goal coordinates")
}

class SpotNextToPerson {
  tool_name "get_spot_next_to_person"
  goal_position_coordinates int[] @description("the goal coordinates")
}


class PullOutOfWater {
  tool_name "special_action"
  description_of_the_action string @description("description of the action you want to take")

}

class Obligation {
  obligation string
}

function MacroActionExecution(agent_position: Position,target_position: Position, obligations: string) -> NavigationAgent | PullOutOfWater {
    client CustomGPT3Turbo
    prompt #"
      {{ _.role("system") }}
      You are a package delivery robot navigating a grid-world.
      You want to navigate to the coordinates [{{ target_position.column }} {{ target_position.row }}].

      You have to fulfill the following obligations. Each obligation is more important than delivering the package:
      {{ obligations }}
      If you want to rescue a drowning person, you need to position yourself at the position at which they need to be rescued, first.
      Otherwise, navigate to the package delivery destination.

      {{ _.role("user") }}
      Make a plan consisting of a sequence of steps where each step is one of the following: 
      1. navigate to the coordinate of the package delivery destination or
      2. navigate to a location where you need to navigate to in order to fulfill an obligation or
      3. fulfill the obligation.
      After making a complete plan, decide which tool you want to use first.

    {{ ctx.output_format }}
    "#
}

test macro_action_execution_test {
  functions [MacroActionExecution]
  args {
    target_position {"row": 0, "column": 0}
    obligations "Rescue the drowning person at position [2,3]."
  }
}
