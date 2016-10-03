require 'Model'

function build_md()

   -- model configuration
  local dimx = 32*32*2
  local dimt = 32*32

  local layers = {
    {type='fc', dimIn=dimx,     dimOut=math.floor(1.5*dimx)}, {type='relu'},
    {type='fc', dimIn=math.floor(1.5*dimx), dimOut=math.floor(3.0*dimx)}, {type='relu'},
    {type='fc', dimIn=math.floor(3.0*dimx), dimOut=math.floor(3.0*dimt)}, {type='relu'},
    {type='fc', dimIn=math.floor(3.0*dimt), dimOut=math.floor(1.5*dimt)}, {type='relu'},
    {type='fc', dimIn=math.floor(1.5*dimt), dimOut=math.floor(dimt)}, {type='relu'},
  }

  return Model(layers)
end


