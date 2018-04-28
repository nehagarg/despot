#include <despot/simple_tui.h>
#include "tiger.h"
#include "tiger2actions.h"

using namespace despot;

class TUI: public SimpleTUI {
public:
  TUI() {
  }
 
  DSPOMDP* InitializeModel(option::Option* options) {
    DSPOMDP* model;
    model = !options[E_PARAMS_FILE] ?
       new  Tiger2actions(): new Tiger();
    return model;
  }
  
  void InitializeDefaultParameters() {
  }
};

int main(int argc, char* argv[]) {
  return TUI().run(argc, argv);
}
