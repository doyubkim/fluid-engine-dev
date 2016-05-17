#include <obj/obj_parser.hpp>

#include <cctype>
#include <fstream>
#include <sstream>

bool obj::obj_parser::parse(std::istream& istream)
{
  std::string line;
  std::size_t line_number = 0;

  std::size_t number_of_geometric_vertices = 0, number_of_texture_vertices = 0, number_of_vertex_normals = 0, number_of_faces = 0, number_of_group_names = 0, number_of_smoothing_groups = 0, number_of_object_names = 0, number_of_material_libraries = 0, number_of_material_names = 0;

  while (!istream.eof() && std::getline(istream, line)) {
    ++line_number;
    std::istringstream stringstream(line);
    stringstream.unsetf(std::ios_base::skipws);

    stringstream >> std::ws;
    if (stringstream.eof()) {
      if (flags_ & parse_blank_lines_as_comment) {
        if (comment_callback_) {
          comment_callback_(line);
        }
      }
    }
    else if (stringstream.peek() == '#') {
      if (comment_callback_) {
        comment_callback_(line);
      }
    }
    else {
      std::string keyword;
      stringstream >> keyword;

      // geometric vertex (v)
      if (keyword == "v") {
        float_type x, y, z;
        char whitespace_v_x, whitespace_x_y, whitespace_y_z;
        stringstream >> whitespace_v_x >> std::ws >> x >> whitespace_x_y >> std::ws >> y >> whitespace_y_z >> std::ws >> z >> std::ws;
        if (!stringstream.eof() || !std::isspace(whitespace_v_x) || !std::isspace(whitespace_x_y) || !std::isspace(whitespace_y_z)) {
          if (error_callback_) {
            error_callback_(line_number, "parse error");
          }
          return false;
        }
        ++number_of_geometric_vertices;
        if (geometric_vertex_callback_) {
          geometric_vertex_callback_(x, y, z);
        }
      }

      // texture vertex (vt)
      else if (keyword == "vt") {
        float_type u, v;
        char whitespace_vt_u, whitespace_u_v;
        stringstream >> whitespace_vt_u >> std::ws >> u >> whitespace_u_v >> std::ws >> v;
        char whitespace_v_w = ' ';
        if (!stringstream.eof()) {
          stringstream >> whitespace_v_w >> std::ws;
        }
        if (!std::isspace(whitespace_vt_u) || !std::isspace(whitespace_u_v) || !std::isspace(whitespace_v_w)) {
          if (error_callback_) {
            error_callback_(line_number, "parse error");
          }
          return false;
        }
        if (stringstream.eof()) {
          ++number_of_texture_vertices;
          if (texture_vertex_callback_) {
            texture_vertex_callback_(u, v);
          }
        }
        else {
          float_type w;
          stringstream >> w >> std::ws;
          if (!stringstream.eof()) {
            if (error_callback_) {
              error_callback_(line_number, "parse error");
            }
            return false;
          }
          ++number_of_texture_vertices;
          if (w == float_type(0.0)) {
            if (texture_vertex_callback_) {
              texture_vertex_callback_(u, v);
            }
          }
          else {
            if (error_callback_) {
              error_callback_(line_number, "parse error");
            }
            return false;
          }
        }
      }

      // vertex normal (vn)
      else if (keyword == "vn") {
        float_type x, y, z;
        char whitespace_vn_x, whitespace_x_y, whitespace_y_z;
        stringstream >> whitespace_vn_x >> std::ws >> x >> whitespace_x_y >> std::ws >> y >> whitespace_y_z >> std::ws >> z >> std::ws;
        if (!stringstream.eof() || !std::isspace(whitespace_vn_x) || !std::isspace(whitespace_x_y) || !std::isspace(whitespace_y_z)) {
          if (error_callback_) {
            error_callback_(line_number, "parse error");
          }
          return false;
        }
        ++number_of_vertex_normals;
        if (vertex_normal_callback_) {
          vertex_normal_callback_(x, y, z);
        }
      }

      // face (f)
      else if ((keyword == "f") || (keyword == "fo")) {
        index_type v1;
        char whitespace_f_v1;
        stringstream >> whitespace_f_v1 >> std::ws >> v1;
        if (std::isspace(stringstream.peek())) {
          // f v
          index_type v2, v3;
          char whitespace_v1_v2, whitespace_v2_v3;
          stringstream >> whitespace_v1_v2 >> std::ws >> v2 >> whitespace_v2_v3 >> std::ws >> v3;
          char whitespace_v3_v4 = ' ';
          if (!stringstream.eof()) {
            stringstream >> whitespace_v3_v4 >> std::ws;
          }
          if (!std::isspace(whitespace_f_v1) || !std::isspace(whitespace_v1_v2) || !std::isspace(whitespace_v2_v3) || !std::isspace(whitespace_v3_v4)) {
            if (error_callback_) {
              error_callback_(line_number, "parse error");
            }
            return false;
          }
          if (((v1 < index_type(-number_of_geometric_vertices)) || (-1 < v1)) && ((v1 < 1) || (index_type(number_of_geometric_vertices) < v1))
           || ((v2 < index_type(-number_of_geometric_vertices)) || (-1 < v2)) && ((v2 < 1) || (index_type(number_of_geometric_vertices) < v2))
           || ((v3 < index_type(-number_of_geometric_vertices)) || (-1 < v3)) && ((v3 < 1) || (index_type(number_of_geometric_vertices) < v3))) {
            if (error_callback_) {
              error_callback_(line_number, "index out of bounds");
            }
            return false;
          }
          if (flags_ & translate_negative_indices) {
            if (v1 < 0) {
              v1 += number_of_geometric_vertices + 1;
            }
            if (v2 < 0) {
              v2 += number_of_geometric_vertices + 1;
            }
            if (v3 < 0) {
              v3 += number_of_geometric_vertices + 1;
            }
          }
          if (stringstream.eof()) {
            ++number_of_faces;
            if (triangular_face_geometric_vertices_callback_) {
              triangular_face_geometric_vertices_callback_(v1, v2, v3);
            }
          }
          else {
            index_type v4;
            stringstream >> v4;
            char whitespace_v4_v5 = ' ';
            if (!stringstream.eof()) {
              stringstream >> whitespace_v4_v5 >> std::ws;
            }
            if (!std::isspace(whitespace_v4_v5)) {
              if (error_callback_) {
                error_callback_(line_number, "parse error");
              }
              return false;
            }
            if (((v4 < index_type(-number_of_geometric_vertices)) || (-1 < v4)) && ((v4 < 1) || (index_type(number_of_geometric_vertices) < v4))) {
              if (error_callback_) {
                error_callback_(line_number, "index out of bounds");
              }
              return false;
            }
            if (flags_ & translate_negative_indices) {
              if (v4 < 0) {
                v4 += number_of_geometric_vertices + 1;
              }
            }
            if (stringstream.eof()) {
              ++number_of_faces;
              if (flags_ & triangulate_faces) {
                if (triangular_face_geometric_vertices_callback_) {
                  triangular_face_geometric_vertices_callback_(v1, v2, v3);
                  triangular_face_geometric_vertices_callback_(v1, v3, v4);
                }
              }
              else {
                if (quadrilateral_face_geometric_vertices_callback_) {
                  quadrilateral_face_geometric_vertices_callback_(v1, v2, v3, v4);
                }
              }
            }
            else {
              if (flags_ & triangulate_faces) {
                if (triangular_face_geometric_vertices_callback_) {
                  triangular_face_geometric_vertices_callback_(v1, v2, v3);
                  triangular_face_geometric_vertices_callback_(v1, v3, v4);
                }
                index_type v_previous = v4;
                do {
                  index_type v;
                  stringstream >> v;
                  char whitespace_v_v = ' ';
                  if (!stringstream.eof()) {
                    stringstream >> whitespace_v_v >> std::ws;
                  }
                  if (stringstream && std::isspace(whitespace_v_v)) {
                    if (((v < index_type(-number_of_geometric_vertices)) || (-1 < v)) && ((v < 1) || (index_type(number_of_geometric_vertices) < v))) {
                      if (error_callback_) {
                        error_callback_(line_number, "index out of bounds");
                      }
                      return false;
                    }
                    if (flags_ & translate_negative_indices) {
                      if (v < 0) {
                        v += number_of_geometric_vertices + 1;
                      }
                    }
                    if (triangular_face_geometric_vertices_callback_) {
                      triangular_face_geometric_vertices_callback_(v1, v_previous, v);
                    }
                    v_previous = v;
                  }
                }
                while (stringstream && !stringstream.eof());
                if (!stringstream.eof()) {
                  if (error_callback_) {
                    error_callback_(line_number, "parse error");
                  }
                  return false;
                }
                ++number_of_faces;
              }
              else {
                if (polygonal_face_geometric_vertices_begin_callback_) {
                  polygonal_face_geometric_vertices_begin_callback_(v1, v2, v3);
                }
                if (polygonal_face_geometric_vertices_vertex_callback_) {
                  polygonal_face_geometric_vertices_vertex_callback_(v4);
                }
                do {
                  index_type v;
                  stringstream >> v;
                  char whitespace_v_v = ' ';
                  if (!stringstream.eof()) {
                    stringstream >> whitespace_v_v >> std::ws;
                  }
                  if (stringstream && std::isspace(whitespace_v_v)) {
                    if (((v < index_type(-number_of_geometric_vertices)) || (-1 < v)) && ((v < 1) || (index_type(number_of_geometric_vertices) < v))) {
                      if (error_callback_) {
                        error_callback_(line_number, "index out of bounds");
                      }
                      return false;
                    }
                    if (flags_ & translate_negative_indices) {
                      if (v < 0) {
                        v += number_of_geometric_vertices + 1;
                      }
                    }
                    if (polygonal_face_geometric_vertices_vertex_callback_) {
                      polygonal_face_geometric_vertices_vertex_callback_(v);
                    }
                  }
                }
                while (stringstream && !stringstream.eof());
                if (!stringstream.eof()) {
                  if (error_callback_) {
                    error_callback_(line_number, "parse error");
                  }
                  return false;
                }
                ++number_of_faces;
                if (polygonal_face_geometric_vertices_end_callback_) {
                  polygonal_face_geometric_vertices_end_callback_();
                }
              }
            }
          }
        }
        else {
          char slash_v1_vt1;
          stringstream >> slash_v1_vt1;
          if (stringstream.peek() != '/') {
            index_type vt1;
            stringstream >> vt1;
            if (std::isspace(stringstream.peek())) {
              // f v/vt
              index_type v2, vt2, v3, vt3;
              char whitespace_vt1_v2, slash_v2_vt2, whitespace_vt2_v3, slash_v3_vt3;
              stringstream >> whitespace_vt1_v2 >> std::ws >> v2 >> slash_v2_vt2 >> vt2 >> whitespace_vt2_v3 >> std::ws >> v3 >> slash_v3_vt3 >> vt3;
              char whitespace_vt3_v4 = ' ';
              if (!stringstream.eof()) {
                stringstream >> whitespace_vt3_v4 >> std::ws;
              }
              if (!std::isspace(whitespace_f_v1) || !(slash_v1_vt1 == '/') || !std::isspace(whitespace_vt1_v2) || !(slash_v2_vt2 == '/') || !std::isspace(whitespace_vt2_v3) || !(slash_v3_vt3 == '/') || !std::isspace(whitespace_vt3_v4)) {
                if (error_callback_) {
                  error_callback_(line_number, "parse error");
                }
                return false;
              }
              if (((v1 < index_type(-number_of_geometric_vertices)) || (-1 < v1)) && ((v1 < 1) || (index_type(number_of_geometric_vertices) < v1))
               || ((vt1 < -index_type(number_of_texture_vertices)) || (-1 < vt1)) && ((vt1 < 1) || (index_type(number_of_texture_vertices) < vt1))
               || ((v2 < index_type(-number_of_geometric_vertices)) || (-1 < v2)) && ((v2 < 1) || (index_type(number_of_geometric_vertices) < v2))
               || ((vt2 < -index_type(number_of_texture_vertices)) || (-1 < vt2)) && ((vt2 < 1) || (index_type(number_of_texture_vertices) < vt2))
               || ((v3 < index_type(-number_of_geometric_vertices)) || (-1 < v3)) && ((v3 < 1) || (index_type(number_of_geometric_vertices) < v3))
               || ((vt3 < -index_type(number_of_texture_vertices)) || (-1 < vt3)) && ((vt3 < 1) || (index_type(number_of_texture_vertices) < vt3))) {
                if (error_callback_) {
                  error_callback_(line_number, "index out of bounds");
                }
                return false;
              }
              if (flags_ & translate_negative_indices) {
                if (v1 < 0) {
                  v1 += number_of_geometric_vertices + 1;
                }
                if (vt1 < 0) {
                  vt1 += number_of_texture_vertices + 1;
                }
                if (v2 < 0) {
                  v2 += number_of_geometric_vertices + 1;
                }
                if (vt2 < 0) {
                  vt2 += number_of_texture_vertices + 1;
                }
                if (v3 < 0) {
                  v3 += number_of_geometric_vertices + 1;
                }
                if (vt3 < 0) {
                  vt3 += number_of_texture_vertices + 1;
                }
              }
              if (stringstream.eof()) {
                ++number_of_faces;
                if (triangular_face_geometric_vertices_texture_vertices_callback_) {
                  triangular_face_geometric_vertices_texture_vertices_callback_(std::make_tuple(v1, vt1), std::make_tuple(v2, vt2), std::make_tuple(v3, vt3));
                }
              }
              else {
                index_type v4, vt4;
                char slash_v4_vt4;
                stringstream >> v4 >> slash_v4_vt4 >> vt4;
                char whitespace_vt4_v5 = ' ';
                if (!stringstream.eof()) {
                  stringstream >> whitespace_vt4_v5 >> std::ws;
                }
                if (!(slash_v4_vt4 == '/') || !std::isspace(whitespace_vt4_v5)) {
                  if (error_callback_) {
                    error_callback_(line_number, "parse error");
                  }
                  return false;
                }
                if (((v4 < index_type(-number_of_geometric_vertices)) || (-1 < v4)) && ((v4 < 1) || (index_type(number_of_geometric_vertices) < v4))
                || ((vt4 < -index_type(number_of_texture_vertices)) || (-1 < vt4)) && ((vt4 < 1) || (index_type(number_of_texture_vertices) < vt4))) {
                  if (error_callback_) {
                    error_callback_(line_number, "index out of bounds");
                  }
                  return false;
                }
                if (flags_ & translate_negative_indices) {
                  if (v4 < 0) {
                    v4 += number_of_geometric_vertices + 1;
                  }
                  if (vt4 < 0) {
                    vt4 += number_of_texture_vertices + 1;
                  }
                }
                if (stringstream.eof()) {
                  ++number_of_faces;
                  if (flags_ & triangulate_faces) {
                    if (triangular_face_geometric_vertices_texture_vertices_callback_) {
                      triangular_face_geometric_vertices_texture_vertices_callback_(std::make_tuple(v1, vt1), std::make_tuple(v2, vt2), std::make_tuple(v3, vt3));
                      triangular_face_geometric_vertices_texture_vertices_callback_(std::make_tuple(v1, vt1), std::make_tuple(v3, vt3), std::make_tuple(v4, vt4));
                    }
                  }
                  else {
                    if (quadrilateral_face_geometric_vertices_texture_vertices_callback_) {
                      quadrilateral_face_geometric_vertices_texture_vertices_callback_(std::make_tuple(v1, vt1), std::make_tuple(v2, vt2), std::make_tuple(v3, vt3), std::make_tuple(v4, vt4));
                    }
                  }
                }
                else {
                  if (flags_ & triangulate_faces) {
                    if (triangular_face_geometric_vertices_texture_vertices_callback_) {
                      triangular_face_geometric_vertices_texture_vertices_callback_(std::make_tuple(v1, vt1), std::make_tuple(v2, vt2), std::make_tuple(v3, vt3));
                      triangular_face_geometric_vertices_texture_vertices_callback_(std::make_tuple(v1, vt1), std::make_tuple(v3, vt3), std::make_tuple(v4, vt4));
                    }
                    index_type v_previous = v4, vt_previous = vt4;
                    do {
                      index_type v, vt;
                      char slash_geometric_vertices_texture_vertices;
                      stringstream >> v >> slash_geometric_vertices_texture_vertices >> vt;
                      char whitespace_vt_v = ' ';
                      if (!stringstream.eof()) {
                        stringstream >> whitespace_vt_v >> std::ws;
                      }
                      if (stringstream && (slash_geometric_vertices_texture_vertices == '/') && std::isspace(whitespace_vt_v)) {
                        if (((v < index_type(-number_of_geometric_vertices)) || (-1 < v)) && ((v < 1) || (index_type(number_of_geometric_vertices) < v))
                        || ((vt < -index_type(number_of_texture_vertices)) || (-1 < vt)) && ((vt < 1) || (index_type(number_of_texture_vertices) < vt))) {
                          if (error_callback_) {
                            error_callback_(line_number, "index out of bounds");
                          }
                          return false;
                        }
                        if (flags_ & translate_negative_indices) {
                          if (v < 0) {
                            v += number_of_geometric_vertices + 1;
                          }
                          if (vt < 0) {
                            vt += number_of_texture_vertices + 1;
                          }
                        }
                        if (triangular_face_geometric_vertices_texture_vertices_callback_) {
                          triangular_face_geometric_vertices_texture_vertices_callback_(std::make_tuple(v1, vt1), std::make_tuple(v_previous, vt_previous), std::make_tuple(v, vt));
                        }
                        v_previous = v, vt_previous = vt;
                      }
                    }
                    while (stringstream && !stringstream.eof());
                    if (!stringstream.eof()) {
                      if (error_callback_) {
                        error_callback_(line_number, "parse error");
                      }
                      return false;
                    }
                    ++number_of_faces;
                  }
                  else {
                    if (polygonal_face_geometric_vertices_texture_vertices_begin_callback_) {
                      polygonal_face_geometric_vertices_texture_vertices_begin_callback_(index_2_tuple_type(v1, vt1), index_2_tuple_type(v2, vt2), index_2_tuple_type(v3, vt3));
                    }
                    if (polygonal_face_geometric_vertices_texture_vertices_vertex_callback_) {
                      polygonal_face_geometric_vertices_texture_vertices_vertex_callback_(index_2_tuple_type(v4, vt4));
                    }
                    do {
                      index_type v, vt;
                      char slash_geometric_vertices_texture_vertices;
                      stringstream >> v >> slash_geometric_vertices_texture_vertices >> vt;
                      char whitespace_vt_v = ' ';
                      if (!stringstream.eof()) {
                        stringstream >> whitespace_vt_v >> std::ws;
                      }
                      if (stringstream && (slash_geometric_vertices_texture_vertices == '/') && std::isspace(whitespace_vt_v)) {
                        if (((v < index_type(-number_of_geometric_vertices)) || (-1 < v)) && ((v < 1) || (index_type(number_of_geometric_vertices) < v))
                        || ((vt < -index_type(number_of_texture_vertices)) || (-1 < vt)) && ((vt < 1) || (index_type(number_of_texture_vertices) < vt))) {
                          if (error_callback_) {
                            error_callback_(line_number, "index out of bounds");
                          }
                          return false;
                        }
                        if (flags_ & translate_negative_indices) {
                          if (v < 0) {
                            v += number_of_geometric_vertices + 1;
                          }
                          if (vt < 0) {
                            vt += number_of_texture_vertices + 1;
                          }
                        }
                        if (polygonal_face_geometric_vertices_texture_vertices_vertex_callback_) {
                          polygonal_face_geometric_vertices_texture_vertices_vertex_callback_(index_2_tuple_type(v, vt));
                        }
                      }
                    }
                    while (stringstream && !stringstream.eof());
                    if (!stringstream.eof()) {
                      if (error_callback_) {
                        error_callback_(line_number, "parse error");
                      }
                      return false;
                    }
                    ++number_of_faces;
                    if (polygonal_face_geometric_vertices_texture_vertices_end_callback_) {
                      polygonal_face_geometric_vertices_texture_vertices_end_callback_();
                    }
                  }
                }
              }
            }
            else {
              // f v/vt/vn
              index_type vn1, v2, vt2, vn2, v3, vt3, vn3;
              char slash_vt1_vn1, whitespace_vn1_v2, slash_v2_vt2, slash_vt2_vn2, whitespace_vn2_v3, slash_v3_vt3, slash_vt3_vn3;
              stringstream >> slash_vt1_vn1 >> vn1 >> whitespace_vn1_v2 >> std::ws >> v2 >> slash_v2_vt2 >> vt2 >> slash_vt2_vn2 >> vn2 >> whitespace_vn2_v3 >> std::ws >> v3 >> slash_v3_vt3 >> vt3 >> slash_vt3_vn3 >> vn3;
              char whitespace_vn3_v4 = ' ';
              if (!stringstream.eof()) {
                stringstream >> whitespace_vn3_v4 >> std::ws;
              }
              if (!std::isspace(whitespace_f_v1) || !(slash_v1_vt1 == '/') || !(slash_vt1_vn1 == '/') || !std::isspace(whitespace_vn1_v2) || !(slash_v2_vt2 == '/') || !(slash_vt2_vn2 == '/') || !std::isspace(whitespace_vn2_v3) || !(slash_v3_vt3 == '/') || !(slash_vt3_vn3 == '/') || !std::isspace(whitespace_vn3_v4)) {
                if (error_callback_) {
                  error_callback_(line_number, "parse error");
                }
                return false;
              }
              if (((v1 < index_type(-number_of_geometric_vertices)) || (-1 < v1)) && ((v1 < 1) || (index_type(number_of_geometric_vertices) < v1))
               || ((vt1 < -index_type(number_of_texture_vertices)) || (-1 < vt1)) && ((vt1 < 1) || (index_type(number_of_texture_vertices) < vt1))
               || ((vn1 < -index_type(number_of_vertex_normals)) || (-1 < vn1)) && ((vn1 < 1) || (index_type(number_of_vertex_normals) < vn1))
               || ((v2 < index_type(-number_of_geometric_vertices)) || (-1 < v2)) && ((v2 < 1) || (index_type(number_of_geometric_vertices) < v2))
               || ((vt2 < -index_type(number_of_texture_vertices)) || (-1 < vt2)) && ((vt2 < 1) || (index_type(number_of_texture_vertices) < vt2))
               || ((vn2 < -index_type(number_of_vertex_normals)) || (-1 < vn2)) && ((vn2 < 1) || (index_type(number_of_vertex_normals) < vn2))
               || ((v3 < index_type(-number_of_geometric_vertices)) || (-1 < v3)) && ((v3 < 1) || (index_type(number_of_geometric_vertices) < v3))
               || ((vt3 < -index_type(number_of_texture_vertices)) || (-1 < vt3)) && ((vt3 < 1) || (index_type(number_of_texture_vertices) < vt3))
               || ((vn3 < -index_type(number_of_vertex_normals)) || (-1 < vn3)) && ((vn3 < 1) || (index_type(number_of_vertex_normals) < vn3))) {
                if (error_callback_) {
                  error_callback_(line_number, "index out of bounds");
                }
                return false;
              }
              if (flags_ & translate_negative_indices) {
                if (v1 < 0) {
                  v1 += number_of_geometric_vertices + 1;
                }
                if (vt1 < 0) {
                  vt1 += number_of_texture_vertices + 1;
                }
                if (vn1 < 0) {
                  vn1 += number_of_vertex_normals + 1;
                }
                if (v2 < 0) {
                  v2 += number_of_geometric_vertices + 1;
                }
                if (vt2 < 0) {
                  vt2 += number_of_texture_vertices + 1;
                }
                if (vn2 < 0) {
                  vn2 += number_of_vertex_normals + 1;
                }
                if (v3 < 0) {
                  v3 += number_of_geometric_vertices + 1;
                }
                if (vt3 < 0) {
                  vt3 += number_of_texture_vertices + 1;
                }
                if (vn3 < 0) {
                  vn3 += number_of_vertex_normals + 1;
                }
              }
              if (stringstream.eof()) {
                ++number_of_faces;
                if (triangular_face_geometric_vertices_texture_vertices_vertex_normals_callback_) {
                  triangular_face_geometric_vertices_texture_vertices_vertex_normals_callback_(std::make_tuple(v1, vt1, vn1), std::make_tuple(v2, vt2, vn2), std::make_tuple(v3, vt3, vn3));
                }
              }
              else {
                index_type v4, vt4, vn4;
                char slash_v4_vt4, slash_vt4_vn4;
                stringstream >> v4 >> slash_v4_vt4 >> vt4 >> slash_vt4_vn4 >> vn4;
                char whitespace_vn4_v5 = ' ';
                if (!stringstream.eof()) {
                  stringstream >> whitespace_vn4_v5 >> std::ws;
                }
                if (!(slash_v4_vt4 == '/') || !(slash_vt4_vn4 == '/') || !std::isspace(whitespace_vn4_v5)) {
                  if (error_callback_) {
                    error_callback_(line_number, "parse error");
                  }
                  return false;
                }
                if (((v4 < index_type(-number_of_geometric_vertices)) || (-1 < v4)) && ((v4 < 1) || (index_type(number_of_geometric_vertices) < v4))
                 || ((vt4 < -index_type(number_of_texture_vertices)) || (-1 < vt4)) && ((vt4 < 1) || (index_type(number_of_texture_vertices) < vt4))
                 || ((vn4 < -index_type(number_of_vertex_normals)) || (-1 < vn4)) && ((vn4 < 1) || (index_type(number_of_vertex_normals) < vn4))) {
                  if (error_callback_) {
                    error_callback_(line_number, "index out of bounds");
                  }
                  return false;
                }
                if (flags_ & translate_negative_indices) {
                  if (v4 < 0) {
                    v4 += number_of_geometric_vertices + 1;
                  }
                  if (vt4 < 0) {
                    vt4 += number_of_texture_vertices + 1;
                  }
                  if (vn4 < 0) {
                    vn4 += number_of_vertex_normals + 1;
                  }
                }
                if (stringstream.eof()) {
                  ++number_of_faces;
                  if (flags_ & triangulate_faces) {
                    if (triangular_face_geometric_vertices_texture_vertices_vertex_normals_callback_) {
                      triangular_face_geometric_vertices_texture_vertices_vertex_normals_callback_(std::make_tuple(v1, vt1, vn1), std::make_tuple(v2, vt2, vn2), std::make_tuple(v3, vt3, vn3));
                      triangular_face_geometric_vertices_texture_vertices_vertex_normals_callback_(std::make_tuple(v1, vt1, vn1), std::make_tuple(v3, vt3, vn3), std::make_tuple(v4, vt4, vn4));
                    }
                  }
                  else {
                    if (quadrilateral_face_geometric_vertices_texture_vertices_vertex_normals_callback_) {
                      quadrilateral_face_geometric_vertices_texture_vertices_vertex_normals_callback_(std::make_tuple(v1, vt1, vn1), std::make_tuple(v2, vt2, vn2), std::make_tuple(v3, vt3, vn3), std::make_tuple(v4, vt4, vn4));
                    }
                  }
                }
                else {
                  if (flags_ & triangulate_faces) {
                    if (triangular_face_geometric_vertices_texture_vertices_vertex_normals_callback_) {
                      triangular_face_geometric_vertices_texture_vertices_vertex_normals_callback_(std::make_tuple(v1, vt1, vn1), std::make_tuple(v2, vt2, vn2), std::make_tuple(v3, vt3, vn3));
                      triangular_face_geometric_vertices_texture_vertices_vertex_normals_callback_(std::make_tuple(v1, vt1, vn1), std::make_tuple(v3, vt3, vn3), std::make_tuple(v4, vt4, vn4));
                    }
                    index_type v_previous = v4, vt_previous = vt4, vn_previous = vn4;
                    do {
                      index_type v, vt, vn;
                      char slash_geometric_vertices_texture_vertices, slash_vt_vn;
                      stringstream >> v >> slash_geometric_vertices_texture_vertices >> vt >> slash_vt_vn >> vn;
                      char whitespace_vn_v = ' ';
                      if (!stringstream.eof()) {
                        stringstream >> whitespace_vn_v >> std::ws;
                      }
                      if (stringstream && (slash_geometric_vertices_texture_vertices == '/') && (slash_vt_vn == '/') && std::isspace(whitespace_vn_v)) {
                        if (((v < index_type(-number_of_geometric_vertices)) || (-1 < v)) && ((v < 1) || (index_type(number_of_geometric_vertices) < v))
                        || ((vt < -index_type(number_of_texture_vertices)) || (-1 < vt)) && ((vt < 1) || (index_type(number_of_texture_vertices) < vt))
                        || ((vn < -index_type(number_of_vertex_normals)) || (-1 < vn)) && ((vn < 1) || (index_type(number_of_vertex_normals) < vn))) {
                          if (error_callback_) {
                            error_callback_(line_number, "index out of bounds");
                          }
                          return false;
                        }
                        if (flags_ & translate_negative_indices) {
                          if (v < 0) {
                            v += number_of_geometric_vertices + 1;
                          }
                          if (vt < 0) {
                            vt += number_of_texture_vertices + 1;
                          }
                          if (vn < 0) {
                            vn += number_of_vertex_normals + 1;
                          }
                        }
                        if (triangular_face_geometric_vertices_texture_vertices_vertex_normals_callback_) {
                          triangular_face_geometric_vertices_texture_vertices_vertex_normals_callback_(std::make_tuple(v1, vt1, vn1), std::make_tuple(v_previous, vt_previous, vn_previous), std::make_tuple(v, vt, vn));
                        }
                        v_previous = v, vt_previous = vt, vn_previous = vn;
                      }
                    }
                    while (stringstream && !stringstream.eof());
                    if (!stringstream.eof()) {
                      if (error_callback_) {
                        error_callback_(line_number, "parse error");
                      }
                      return false;
                    }
                    ++number_of_faces;
                  }
                  else {
                    if (polygonal_face_geometric_vertices_texture_vertices_vertex_normals_begin_callback_) {
                      polygonal_face_geometric_vertices_texture_vertices_vertex_normals_begin_callback_(index_3_tuple_type(v1, vt1, vn1), index_3_tuple_type(v2, vt2, vn2), index_3_tuple_type(v3, vt3, vn3));
                    }
                    if (polygonal_face_geometric_vertices_texture_vertices_vertex_normals_vertex_callback_) {
                      polygonal_face_geometric_vertices_texture_vertices_vertex_normals_vertex_callback_(index_3_tuple_type(v4, vt4, vn4));
                    }
                    do {
                      index_type v, vt, vn;
                      char slash_geometric_vertices_texture_vertices, slash_vt_vn;
                      stringstream >> v >> slash_geometric_vertices_texture_vertices >> vt >> slash_vt_vn >> vn;
                      char whitespace_vn_v = ' ';
                      if (!stringstream.eof()) {
                        stringstream >> whitespace_vn_v >> std::ws;
                      }
                      if (stringstream && (slash_geometric_vertices_texture_vertices == '/') && (slash_vt_vn == '/') && std::isspace(whitespace_vn_v)) {
                        if (((v < index_type(-number_of_geometric_vertices)) || (-1 < v)) && ((v < 1) || (index_type(number_of_geometric_vertices) < v))
                        || ((vt < -index_type(number_of_texture_vertices)) || (-1 < vt)) && ((vt < 1) || (index_type(number_of_texture_vertices) < vt))
                        || ((vn < -index_type(number_of_vertex_normals)) || (-1 < vn)) && ((vn < 1) || (index_type(number_of_vertex_normals) < vn))) {
                          if (error_callback_) {
                            error_callback_(line_number, "index out of bounds");
                          }
                          return false;
                        }
                        if (flags_ & translate_negative_indices) {
                          if (v < 0) {
                            v += number_of_geometric_vertices + 1;
                          }
                          if (vt < 0) {
                            vt += number_of_texture_vertices + 1;
                          }
                          if (vn < 0) {
                            vn += number_of_vertex_normals + 1;
                          }
                        }
                        if (polygonal_face_geometric_vertices_texture_vertices_vertex_normals_vertex_callback_) {
                          polygonal_face_geometric_vertices_texture_vertices_vertex_normals_vertex_callback_(index_3_tuple_type(v, vt, vn));
                        }
                      }
                    }
                    while (stringstream && !stringstream.eof());
                    if (!stringstream.eof()) {
                      if (error_callback_) {
                        error_callback_(line_number, "parse error");
                      }
                      return false;
                    }
                    ++number_of_faces;
                    if (polygonal_face_geometric_vertices_texture_vertices_vertex_normals_end_callback_) {
                      polygonal_face_geometric_vertices_texture_vertices_vertex_normals_end_callback_();
                    }
                  }
                }
              }
            }
          }
          else {
            // f v//vn
            index_type vn1, v2, vn2, v3, vn3;
            char slash_vt1_vn1, whitespace_vn1_v2, slash_v2_vt2, slash_vt2_vn2, whitespace_vn2_v3, slash_v3_vt3, slash_vt3_vn3;
            stringstream >> slash_vt1_vn1 >> vn1 >> whitespace_vn1_v2 >> std::ws >> v2 >> slash_v2_vt2 >> slash_vt2_vn2 >> vn2 >> whitespace_vn2_v3 >> std::ws >> v3 >> slash_v3_vt3 >> slash_vt3_vn3 >> vn3;
            char whitespace_vn3_v4 = ' ';
            if (!stringstream.eof()) {
              stringstream >> whitespace_vn3_v4 >> std::ws;
            }
            if (!std::isspace(whitespace_f_v1) || !(slash_v1_vt1 == '/') || !(slash_vt1_vn1 == '/') || !std::isspace(whitespace_vn1_v2) || !(slash_v2_vt2 == '/') || !(slash_vt2_vn2 == '/') || !std::isspace(whitespace_vn2_v3) || !(slash_v3_vt3 == '/') || !(slash_vt3_vn3 == '/') || !std::isspace(whitespace_vn3_v4)) {
              if (error_callback_) {
                error_callback_(line_number, "parse error");
              }
              return false;
            }
            if (((v1 < index_type(-number_of_geometric_vertices)) || (-1 < v1)) && ((v1 < 1) || (index_type(number_of_geometric_vertices) < v1))
              || ((vn1 < -index_type(number_of_vertex_normals)) || (-1 < vn1)) && ((vn1 < 1) || (index_type(number_of_vertex_normals) < vn1))
              || ((v2 < index_type(-number_of_geometric_vertices)) || (-1 < v2)) && ((v2 < 1) || (index_type(number_of_geometric_vertices) < v2))
              || ((vn2 < -index_type(number_of_vertex_normals)) || (-1 < vn2)) && ((vn2 < 1) || (index_type(number_of_vertex_normals) < vn2))
              || ((v3 < index_type(-number_of_geometric_vertices)) || (-1 < v3)) && ((v3 < 1) || (index_type(number_of_geometric_vertices) < v3))
              || ((vn3 < -index_type(number_of_vertex_normals)) || (-1 < vn3)) && ((vn3 < 1) || (index_type(number_of_vertex_normals) < vn3))) {
              if (error_callback_) {
                error_callback_(line_number, "index out of bounds");
              }
              return false;
            }
            if (flags_ & translate_negative_indices) {
              if (v1 < 0) {
                v1 += number_of_geometric_vertices + 1;
              }
              if (vn1 < 0) {
                vn1 += number_of_vertex_normals + 1;
              }
              if (v2 < 0) {
                v2 += number_of_geometric_vertices + 1;
              }
              if (vn2 < 0) {
                vn2 += number_of_vertex_normals + 1;
              }
              if (v3 < 0) {
                v3 += number_of_geometric_vertices + 1;
              }
              if (vn3 < 0) {
                vn3 += number_of_vertex_normals + 1;
              }
            }
            if (stringstream.eof()) {
              ++number_of_faces;
              if (triangular_face_geometric_vertices_vertex_normals_callback_) {
                triangular_face_geometric_vertices_vertex_normals_callback_(std::make_tuple(v1, vn1), std::make_tuple(v2, vn2), std::make_tuple(v3, vn3));
              }
            }
            else {
              index_type v4, vn4;
              char slash_v4_vt4, slash_vt4_vn4;
              stringstream >> v4 >> slash_v4_vt4 >> slash_vt4_vn4 >> vn4;
              char whitespace_vn4_v5 = ' ';
              if (!stringstream.eof()) {
                stringstream >> whitespace_vn4_v5 >> std::ws;
              }
              if (!(slash_v4_vt4 == '/') || !(slash_vt4_vn4 == '/') || !std::isspace(whitespace_vn4_v5)) {
                if (error_callback_) {
                  error_callback_(line_number, "parse error");
                }
                return false;
              }
              if (((v4 < index_type(-number_of_geometric_vertices)) || (-1 < v4)) && ((v4 < 1) || (index_type(number_of_geometric_vertices) < v4))
                || ((vn4 < -index_type(number_of_vertex_normals)) || (-1 < vn4)) && ((vn4 < 1) || (index_type(number_of_vertex_normals) < vn4))) {
                if (error_callback_) {
                  error_callback_(line_number, "index out of bounds");
                }
                return false;
              }
              if (flags_ & translate_negative_indices) {
                if (v4 < 0) {
                  v4 += number_of_geometric_vertices + 1;
                }
                if (vn4 < 0) {
                  vn4 += number_of_vertex_normals + 1;
                }
              }
              if (stringstream.eof()) {
                ++number_of_faces;
                if (flags_ & triangulate_faces) {
                  if (triangular_face_geometric_vertices_vertex_normals_callback_) {
                    triangular_face_geometric_vertices_vertex_normals_callback_(std::make_tuple(v1, vn1), std::make_tuple(v2, vn2), std::make_tuple(v3, vn3));
                    triangular_face_geometric_vertices_vertex_normals_callback_(std::make_tuple(v1, vn1), std::make_tuple(v3, vn3), std::make_tuple(v4, vn4));
                  }
                }
                else {
                  if (quadrilateral_face_geometric_vertices_vertex_normals_callback_) {
                    quadrilateral_face_geometric_vertices_vertex_normals_callback_(std::make_tuple(v1, vn1), std::make_tuple(v2, vn2), std::make_tuple(v3, vn3), std::make_tuple(v4, vn4));
                  }
                }
              }
              else {
                if (flags_ & triangulate_faces) {
                  if (triangular_face_geometric_vertices_vertex_normals_callback_) {
                    triangular_face_geometric_vertices_vertex_normals_callback_(std::make_tuple(v1, vn1), std::make_tuple(v2, vn2), std::make_tuple(v3, vn3));
                    triangular_face_geometric_vertices_vertex_normals_callback_(std::make_tuple(v1, vn1), std::make_tuple(v3, vn3), std::make_tuple(v4, vn4));
                  }
                  index_type v_previous = v4, vn_previous = vn4;
                  do {
                    index_type v, vn;
                    char slash_geometric_vertices_texture_vertices, slash_vt_vn;
                    stringstream >> v >> slash_geometric_vertices_texture_vertices >> slash_vt_vn >> vn;
                    char whitespace_vn_v = ' ';
                    if (!stringstream.eof()) {
                      stringstream >> whitespace_vn_v >> std::ws;
                    }
                    if (stringstream && (slash_geometric_vertices_texture_vertices == '/') && (slash_vt_vn == '/') && std::isspace(whitespace_vn_v)) {
                      if (((v < index_type(-number_of_geometric_vertices)) || (-1 < v)) && ((v < 1) || (index_type(number_of_geometric_vertices) < v))
                        || ((vn < -index_type(number_of_vertex_normals)) || (-1 < vn)) && ((vn < 1) || (index_type(number_of_vertex_normals) < vn))) {
                        if (error_callback_) {
                          error_callback_(line_number, "index out of bounds");
                        }
                        return false;
                      }
                      if (flags_ & translate_negative_indices) {
                        if (v < 0) {
                          v += number_of_geometric_vertices + 1;
                        }
                        if (vn < 0) {
                          vn += number_of_vertex_normals + 1;
                        }
                      }
                      if (triangular_face_geometric_vertices_vertex_normals_callback_) {
                        triangular_face_geometric_vertices_vertex_normals_callback_(std::make_tuple(v1, vn1), std::make_tuple(v_previous, vn_previous), std::make_tuple(v, vn));
                      }
                      v_previous = v, vn_previous = vn;
                    }
                  }
                  while (stringstream && !stringstream.eof());
                  if (!stringstream.eof()) {
                    if (error_callback_) {
                      error_callback_(line_number, "parse error");
                    }
                    return false;
                  }
                  ++number_of_faces;
                }
                else {
                  if (polygonal_face_geometric_vertices_vertex_normals_begin_callback_) {
                    polygonal_face_geometric_vertices_vertex_normals_begin_callback_(index_2_tuple_type(v1, vn1), index_2_tuple_type(v2, vn2), index_2_tuple_type(v3, vn3));
                  }
                  if (polygonal_face_geometric_vertices_vertex_normals_vertex_callback_) {
                    polygonal_face_geometric_vertices_vertex_normals_vertex_callback_(index_2_tuple_type(v4, vn4));
                  }
                  do {
                    index_type v, vn;
                    char slash_geometric_vertices_texture_vertices, slash_vt_vn;
                    stringstream >> v >> slash_geometric_vertices_texture_vertices >> slash_vt_vn >> vn;
                    char whitespace_vn_v = ' ';
                    if (!stringstream.eof()) {
                      stringstream >> whitespace_vn_v >> std::ws;
                    }
                    if (stringstream && (slash_geometric_vertices_texture_vertices == '/') && (slash_vt_vn == '/') && std::isspace(whitespace_vn_v)) {
                      if (((v < index_type(-number_of_geometric_vertices)) || (-1 < v)) && ((v < 1) || (index_type(number_of_geometric_vertices) < v))
                        || ((vn < index_type(-number_of_vertex_normals)) || (-1 < vn)) && ((vn < 1) || (index_type(number_of_vertex_normals) < vn))) {
                        if (error_callback_) {
                          error_callback_(line_number, "index out of bounds");
                        }
                        return false;
                      }
                      if (flags_ & translate_negative_indices) {
                        if (v < 0) {
                          v += number_of_geometric_vertices + 1;
                        }
                        if (vn < 0) {
                          vn += number_of_vertex_normals + 1;
                        }
                      }
                      if (polygonal_face_geometric_vertices_vertex_normals_vertex_callback_) {
                        polygonal_face_geometric_vertices_vertex_normals_vertex_callback_(index_2_tuple_type(v, vn));
                      }
                    }
                  }
                  while (stringstream && !stringstream.eof());
                  if (!stringstream.eof()) {
                    if (error_callback_) {
                      error_callback_(line_number, "parse error");
                    }
                    return false;
                  }
                  ++number_of_faces;
                  if (polygonal_face_geometric_vertices_vertex_normals_end_callback_) {
                    polygonal_face_geometric_vertices_vertex_normals_end_callback_();
                  }
                }
              }
            }
          }
        }
      }

      // group name (g)
      else if (keyword == "g") {
        char whitespace_mtllib_group_name = ' ';
        if (!stringstream.eof()) {
          stringstream >> whitespace_mtllib_group_name >> std::ws;
        }
        if (!std::isspace(whitespace_mtllib_group_name)) {
          if (error_callback_) {
            error_callback_(line_number, "parse error");
          }
          return false;
        }
        if (stringstream.eof()) {
          ++number_of_group_names;
          if (group_name_callback_) {
            group_name_callback_("default");
          }
        }
        else {
          std::string group_name;
          stringstream >> group_name >> std::ws;
          if (!stringstream.eof()) {
            if (error_callback_) {
              error_callback_(line_number, "parse error");
            }
            return false;
          }
          ++number_of_group_names;
          if (group_name_callback_) {
            group_name_callback_(group_name);
          }
        }
      }

      // smoothing group (s)
      else if (keyword == "s") {
        std::string group_number_string;
        char whitespace_mtllib_group_number;
        stringstream >> whitespace_mtllib_group_number >> std::ws >> group_number_string >> std::ws;
        if (!stringstream.eof() || !std::isspace(whitespace_mtllib_group_number)) {
          if (error_callback_) {
            error_callback_(line_number, "parse error");
          }
          return false;
        }
        size_type group_number;
        if (group_number_string == "off") {
          group_number = 0;
        }
        else {
          std::istringstream stringstream(group_number_string);
          stringstream >> group_number;
          if (!stringstream.eof()) {
            if (error_callback_) {
              error_callback_(line_number, "parse error");
            }
            return false;
          }
        }
        ++number_of_smoothing_groups;
        if (smoothing_group_callback_) {
          smoothing_group_callback_(group_number);
        }
      }

      // object name (o)
      else if (keyword == "o") {
        std::string object_name;
        char whitespace_mtllib_object_name;
        stringstream >> whitespace_mtllib_object_name >> std::ws >> object_name >> std::ws;
        if (!stringstream.eof() || !std::isspace(whitespace_mtllib_object_name)) {
          if (error_callback_) {
            error_callback_(line_number, "parse error");
          }
          return false;
        }
        ++number_of_object_names;
        if (object_name_callback_) {
          object_name_callback_(object_name);
        }
      }

      // material library (mtllib)
      else if (keyword == "mtllib") {
        std::string filename;
        char whitespace_mtllib_filename;
        stringstream >> whitespace_mtllib_filename >> std::ws >> filename >> std::ws;
        if (!stringstream.eof() || !std::isspace(whitespace_mtllib_filename)) {
          if (error_callback_) {
            error_callback_(line_number, "parse error");
          }
          return false;
        }
        ++number_of_material_libraries;
        if (material_library_callback_) {
          material_library_callback_(filename);
        }
      }

      // material name (usemtl)
      else if (keyword == "usemtl") {
        std::string material_name;
        char whitespace_mtllib_material_name;
        stringstream >> whitespace_mtllib_material_name >> std::ws >> material_name >> std::ws;
        if (!stringstream.eof() || !std::isspace(whitespace_mtllib_material_name)) {
          if (error_callback_) {
            error_callback_(line_number, "parse error");
          }
          return false;
        }
        ++number_of_material_names;
        if (material_name_callback_) {
          material_name_callback_(material_name);
        }
      }

      // unknown keyword
      else {
        std::string message = "ignoring line '" + line + "'";
        if (warning_callback_) {
          warning_callback_(line_number, message);
        }
      }

    }
  }

  return istream.fail() && istream.eof() && !istream.bad();
}
