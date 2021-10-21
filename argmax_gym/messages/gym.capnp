@0xf19aa837ee7ec17c;

using import "image.capnp".ImageProto;

struct GymReturnProto {
  observation @0 :List(Float32);
  reward @1 :Float32;
  done @2 :Bool;
  image @3 :ImageProto;
  imageBuffer @4 :List(UInt8);
}

struct GymControlProto {
  action @0 :List(Float32);
  reset @1 :Bool;
  render @2 :Bool;
  done @3 :Bool;
}

