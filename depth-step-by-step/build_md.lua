require 'Model'

function build_md()

   -- model configuration
  local dimx = 32*32
  local dimt = 32*32

  local layers = {
    {type='fc', dimIn=dimx, dimOut=1}, {type='relu'},
    {type='fc', dimIn=1, dimOut=dimt}, {type='relu'}
  }

  return Model(layers)
end


